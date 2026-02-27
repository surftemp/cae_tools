"""
Joint UNET + Flow CN v3 Trainer — Flow-Only with Learned Loss Balancing

Key changes from previous version:
  1. Flow-only: No score/GMM code paths.
  2. Learned homoscedastic uncertainty weighting (Kendall & Gal 2018):
     L_total = exp(-s_mse) * L_mse + s_mse
            + exp(-s_flow) * L_flow_shifted + s_flow
     where L_flow_shifted = max(l_cn_own + C, epsilon)
     and C = 2 * |final_pretrain_nll| (fixed constant set at transition).
     The shift is needed because Kendall requires positive losses,
     but flow NLL on continuous data is negative. The shift does not
     affect CN gradients (dC/d(params) = 0). The floor epsilon = 0.5
     prevents the Kendall weight from diverging.
  3. Adaptive pre-training: CN trains until NLL convergence detected
     (epoch-over-epoch change < threshold for N consecutive epochs),
     then transitions to joint training automatically.
  4. Parallel phase: CN and UNet train independently (no gradient flow).
     UNet trains on MSE vs raw targets so it is warm at joint transition.
  5. Hard gating in CN: clean pixels pass through exactly.

Standing instruction: no VGG/perceptual loss code.
"""

import os
import time
import json
import signal
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from cae_tools.models.unet import UNET, augment_batch
from cae_tools.models.correction_network_v3 import (
    FlowCorrectionNetwork,
    CorrectionNetworkBase,
)

ERA5_CHANNEL_IDX = 3
# Static-only channels fed to CN — no monthly-derived channels
# land_cover=0, elevation=2, slope_magnitude=6, slope_direction=7
CN_STATIC_CHANNEL_INDICES = [0, 2, 6, 7]
EMA_ALPHA = 0.3
RATIO_WARMUP_EPOCHS = 50
N_ERA5_BINS = 10
SHIFT_FLOOR = 0.5  # Floor for shifted flow loss to prevent Kendall weight divergence


def compute_subgroup_robustness_loss(pred, target, inputs, n_era5_bins=N_ERA5_BINS,
                                     min_samples=50):
    era5 = inputs[:, ERA5_CHANNEL_IDX:ERA5_CHANNEL_IDX + 1, :, :]
    se = (pred - target).pow(2)
    era5_flat = era5.reshape(-1)
    se_flat = se.reshape(-1)

    quantiles = torch.linspace(0, 1, n_era5_bins + 1, device=era5.device)
    bin_edges = torch.quantile(era5_flat, quantiles)

    subgroup_mses = []
    for i in range(n_era5_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_era5_bins - 1:
            mask = (era5_flat >= lo) & (era5_flat < hi)
        else:
            mask = (era5_flat >= lo) & (era5_flat <= hi)
        n = mask.sum()
        if n >= min_samples:
            subgroup_mses.append(se_flat[mask].mean())

    if len(subgroup_mses) < 2:
        return torch.tensor(0.0, device=pred.device), {}

    subgroup_mses = torch.stack(subgroup_mses)
    # Use variance, not std. std = sqrt(var) has gradient 1/(2*sqrt(var)) → ∞
    # as subgroup MSEs equalise (the optimum). var has gradient 2(x-mean)/N → 0.
    robustness_loss = subgroup_mses.var()
    info = {f'mse_era5_bin_{i}': m.item() for i, m in enumerate(subgroup_mses)}
    info['n_subgroups'] = len(subgroup_mses)
    info['mse_var_across_bins'] = robustness_loss.item()
    return robustness_loss, info


class JointUNETv3:
    """
    Joint UNET + Flow CN v3 trainer.

    Learned loss balancing (homoscedastic uncertainty weighting):
        L = exp(-s_mse) * L_mse + s_mse
          + exp(-s_flow) * max(L_cn_own + C, eps) + s_flow
          + L_pearson + L_cold + L_subgroup

    C = 2 * |final_pretrain_nll|, set once at pre-train→joint transition.
    eps = 0.5 floor to prevent Kendall weight divergence.

    The log-variance params s_mse and s_flow are learned via gradient
    descent alongside the model weights.
    """

    def __init__(
        self,
        unet,
        cn_base_channels=32,
        cn_dropout_rate=0.0,
        cn_lr=0.0001,
        # Adaptive pre-training
        cn_pretrain_min_epochs=10,
        cn_pretrain_max_epochs=200,
        cn_convergence_threshold=0.01,
        cn_convergence_window=5,
        # Loss terms
        lambda_cold=0.1,
        cold_threshold_k=10.0,
        lambda_subgroup=0.1,
        # Flow CN specific
        flow_n_coupling_layers=4,
        flow_coupling_hidden=32,
        flow_nll_threshold=5.0,
    ):
        self.unet = unet
        self.cn_lr = cn_lr

        # Adaptive pre-training params
        self.cn_pretrain_min_epochs = cn_pretrain_min_epochs
        self.cn_pretrain_max_epochs = cn_pretrain_max_epochs
        self.cn_convergence_threshold = cn_convergence_threshold
        self.cn_convergence_window = cn_convergence_window

        self.lambda_cold = lambda_cold
        self.cold_threshold_k = cold_threshold_k
        self.lambda_subgroup = lambda_subgroup

        # Build flow CN
        self.cn = FlowCorrectionNetwork(
            static_channel_indices=CN_STATIC_CHANNEL_INDICES,
            base_channels=cn_base_channels,
            n_coupling_layers=flow_n_coupling_layers,
            coupling_hidden=flow_coupling_hidden,
            nll_correction_threshold=flow_nll_threshold,
            dropout_rate=cn_dropout_rate,
        )

        # Learned loss balancing parameters (Kendall & Gal 2018)
        # Wrapped in nn.Module so .to(device) works correctly
        self.loss_weights = nn.Module()
        self.loss_weights.log_var_mse = nn.Parameter(torch.tensor(0.0))
        self.loss_weights.log_var_flow = nn.Parameter(torch.tensor(0.0))

        # NLL shift constant: set at pre-train→joint transition as
        # C = 2 * |final_pretrain_nll|. Ensures l_cn_own + C > 0
        # so Kendall weighting has a finite equilibrium.
        self.nll_shift_C = None  # Set during training

    def _get_era5(self, inputs):
        return inputs[:, ERA5_CHANNEL_IDX:ERA5_CHANNEL_IDX + 1, :, :]

    def _unet_forward(self, inputs):
        if self.unet.architecture == 'flow_matching':
            device = next(self.unet.flow_model.parameters()).device
            B = inputs.shape[0]
            t = torch.ones(B, device=device)
            x_t = torch.zeros(B, self.unet.output_shape[0],
                              inputs.shape[2], inputs.shape[3], device=device)
            return self.unet.flow_model(x_t, inputs, t)
        else:
            enc_out, skips = self.unet.encoder(inputs)
            return self.unet.decoder(enc_out, skips)

    def _hard_cold_pct(self, pred_norm, era5_norm):
        norm_params = self.unet.normalisation_parameters
        if isinstance(norm_params, list):
            min_out = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
            era5_min = norm_params[0]['era5_skt']
            era5_max = norm_params[1]['era5_skt']
        else:
            min_out = norm_params['min_output']
            max_out = norm_params['max_output']
            era5_min = norm_params['min_inputs']['era5_skt']
            era5_max = norm_params['max_inputs']['era5_skt']
        pred_k = pred_norm.detach() * (max_out - min_out) + min_out
        era5_k = era5_norm.detach() * (era5_max - era5_min) + era5_min
        cold = (pred_k < (era5_k - self.cold_threshold_k)).float().mean().item()
        return cold * 100.0

    def _soft_cold_pixel_rate(self, pred, era5_norm):
        norm_params = self.unet.normalisation_parameters
        if isinstance(norm_params, list):
            min_out = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
            era5_min = norm_params[0]['era5_skt']
            era5_max = norm_params[1]['era5_skt']
        else:
            min_out = norm_params['min_output']
            max_out = norm_params['max_output']
            era5_min = norm_params['min_inputs']['era5_skt']
            era5_max = norm_params['max_inputs']['era5_skt']
        pred_k = pred * (max_out - min_out) + min_out
        era5_k = era5_norm * (era5_max - era5_min) + era5_min
        threshold_k = era5_k - self.cold_threshold_k
        soft_cold = torch.sigmoid(2.0 * (threshold_k - pred_k))
        return soft_cold.mean()

    def _check_pretrain_convergence(self, nll_history):
        """Check if flow NLL has converged during pre-training."""
        if len(nll_history) < self.cn_convergence_window + 1:
            return False
        recent_changes = [
            abs(nll_history[-i] - nll_history[-i - 1])
            for i in range(1, self.cn_convergence_window + 1)
        ]
        return all(c < self.cn_convergence_threshold for c in recent_changes)

    def train_joint(self, train_ds, test_ds, model_folder, nr_epochs, batch_size,
                    checkpoint_interval=None, database_path=None,
                    training_paths="", test_paths=""):

        # ---- Setup UNET internals ----
        self.unet.set_input_spec(train_ds.get_input_spec())
        self.unet.set_output_spec(train_ds.get_output_spec())
        self.unet.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds.set_normalisation_parameters(self.unet.normalisation_parameters)

        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()
        self.unet.input_shape = (input_chan, input_y, input_x)
        self.unet.output_shape = (output_chan, output_y, output_x)

        from cae_tools.models.model_sizer import create_model_spec
        if not self.unet.spec:
            self.unet.spec = create_model_spec(
                input_size=(input_y, input_x), input_channels=input_chan,
                output_size=(output_y, output_x), output_channels=output_chan,
                kernel_size=self.unet.conv_kernel_size, stride=self.unet.conv_stride,
                input_layer_count=self.unet.conv_input_layer_count,
                output_layer_count=self.unet.conv_output_layer_count
            )

        from cae_tools.models.standard_unet import StandardEncoder, StandardDecoder
        from cae_tools.models.flow_matching_unet import FlowMatchingUNet
        from cae_tools.models.unet import Encoder, Decoder

        if self.unet.architecture == 'flow_matching':
            if not self.unet.flow_model:
                self.unet.flow_model = FlowMatchingUNet(
                    cond_channels=input_chan, target_channels=output_chan,
                    base_channels=self.unet.base_channels,
                    dropout_rate=self.unet.dropout_rate)
        else:
            if not self.unet.encoder:
                if self.unet.architecture == 'standard':
                    self.unet.encoder = StandardEncoder(
                        in_channels=input_chan, base_channels=self.unet.base_channels,
                        dropout_rate=self.unet.dropout_rate)
                else:
                    use_fc = (self.unet.bottleneck_type == 'fc')
                    self.unet.encoder = Encoder(
                        self.unet.spec.get_input_layers(),
                        encoded_space_dim=self.unet.encoded_dim_size,
                        fc_size=self.unet.fc_size,
                        dropout_rate=self.unet.dropout_rate,
                        use_fc=use_fc,
                        latent_activation=self.unet.latent_activation)
            if not self.unet.decoder:
                if self.unet.architecture == 'standard':
                    self.unet.decoder = StandardDecoder(
                        out_channels=output_chan, base_channels=self.unet.base_channels,
                        dropout_rate=self.unet.dropout_rate,
                        output_activation=self.unet.output_activation)
                else:
                    use_fc = (self.unet.bottleneck_type == 'fc')
                    self.unet.decoder = Decoder(
                        self.unet.spec.get_output_layers(),
                        encoded_space_dim=self.unet.encoded_dim_size,
                        fc_size=self.unet.fc_size,
                        dropout_rate=self.unet.dropout_rate,
                        use_fc=use_fc,
                        use_attention=self.unet.use_attention,
                        skip_mode=self.unet.skip_mode,
                        skip_dropout=self.unet.skip_dropout,
                        skip_scale=self.unet.skip_scale,
                        latent_activation=self.unet.latent_activation,
                        output_activation=self.unet.output_activation)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {device}")

        if self.unet.architecture == 'flow_matching':
            self.unet.flow_model.to(device)
        else:
            self.unet.encoder.to(device)
            self.unet.decoder.to(device)
        self.cn = self.cn.to(device)
        self.loss_weights = self.loss_weights.to(device)

        os.makedirs(model_folder, exist_ok=True)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True)

        if self.unet.architecture == 'flow_matching':
            unet_params = list(self.unet.flow_model.parameters())
        else:
            unet_params = (list(self.unet.encoder.parameters()) +
                           list(self.unet.decoder.parameters()))

        unet_optimizer = torch.optim.AdamW(
            unet_params, lr=self.unet.lr, weight_decay=self.unet.weight_decay)

        # CN optimizer includes: CN parameters + loss weighting parameters
        cn_optimizer = torch.optim.AdamW(
            list(self.cn.parameters()) + list(self.loss_weights.parameters()),
            lr=self.cn_lr, weight_decay=self.unet.weight_decay)

        epochs_already_done = self.unet.history.get('nr_epochs', 0)
        if 'total_nr_epochs' not in self.unet.history:
            self.unet.history['total_nr_epochs'] = nr_epochs
        T_max = self.unet.history['total_nr_epochs']
        epochs_this_job = nr_epochs

        # Restore pre-training state
        pretrain_converged = self.unet.history.get('_cn_pretrain_converged', False)
        cn_pretrain_done_epoch = self.unet.history.get(
            '_cn_pretrain_done_epoch', self.cn_pretrain_max_epochs)
        nll_history = self.unet.history.get('_nll_history', [])

        # Restore NLL shift constant if previously set
        if self.nll_shift_C is None:
            self.nll_shift_C = self.unet.history.get('_nll_shift_C', None)

        print(f"Scheduler: T_max={T_max}, already done={epochs_already_done}, "
              f"this job={epochs_this_job}")
        print(f"CN type: flow, adaptive parallel phase "
              f"(min={self.cn_pretrain_min_epochs}, max={self.cn_pretrain_max_epochs}, "
              f"conv_thresh={self.cn_convergence_threshold}, "
              f"conv_window={self.cn_convergence_window})")
        print(f"Learned loss balancing: log_var_mse={self.loss_weights.log_var_mse.item():.3f}, "
              f"log_var_flow={self.loss_weights.log_var_flow.item():.3f}")
        if pretrain_converged:
            print(f"Parallel phase already ended at epoch {cn_pretrain_done_epoch}")
            if self.nll_shift_C is not None:
                print(f"NLL shift constant C = {self.nll_shift_C:.4f}")

        unet_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            unet_optimizer, T_max=T_max, eta_min=1e-5)
        cn_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            cn_optimizer, T_max=T_max, eta_min=1e-6)
        self.unet._scheduler = unet_scheduler

        # Restore optimizer/scheduler states
        if getattr(self.unet, '_saved_optimizer_state_path', None):
            unet_optimizer.load_state_dict(
                torch.load(self.unet._saved_optimizer_state_path, map_location=device))
            self.unet._saved_optimizer_state_path = None
        if getattr(self.unet, '_saved_scheduler_state_path', None):
            unet_scheduler.load_state_dict(
                torch.load(self.unet._saved_scheduler_state_path, map_location=device))
            self.unet._saved_scheduler_state_path = None

        joint_state_dir = os.path.join(model_folder, 'joint_state')
        cn_opt_path = os.path.join(joint_state_dir, 'cn_optimizer.state')
        cn_sch_path = os.path.join(joint_state_dir, 'cn_scheduler.state')
        log_var_path = os.path.join(joint_state_dir, 'log_var.pt')
        if os.path.exists(cn_opt_path):
            cn_optimizer.load_state_dict(torch.load(cn_opt_path, map_location=device))
        if os.path.exists(cn_sch_path):
            cn_scheduler.load_state_dict(torch.load(cn_sch_path, map_location=device))
        if os.path.exists(log_var_path):
            lv = torch.load(log_var_path, map_location=device)
            self.loss_weights.log_var_mse.data = lv['log_var_mse']
            self.loss_weights.log_var_flow.data = lv['log_var_flow']
            print(f"Restored log_var_mse={self.loss_weights.log_var_mse.item():.3f}, "
                  f"log_var_flow={self.loss_weights.log_var_flow.item():.3f}")

        scaler = GradScaler()

        _ema_test = self.unet.history.get('_ema_test_mse', None)
        _best_ema_test = self.unet.history.get('_best_ema_test_mse', float('inf'))
        _ema_ratio = self.unet.history.get('_ema_ratio', None)
        _best_ema_ratio = self.unet.history.get('_best_train_test_ratio', float('inf'))

        self.unet._sigterm_received = False
        def _handle_sigterm(signum, frame):
            print("\n[SIGTERM received] finishing current epoch then saving...")
            self.unet._sigterm_received = True
        signal.signal(signal.SIGTERM, _handle_sigterm)

        self.unet.loss_fn = nn.MSELoss()
        start = time.time()

        cn_params_count = sum(p.numel() for p in self.cn.parameters())
        print(f"CN parameters: {cn_params_count:,} ({cn_params_count/1e6:.2f}M)")

        # ================================================================
        # TRAINING LOOP
        # ================================================================
        try:
            for epoch in range(epochs_this_job):
                global_epoch = epochs_already_done + epoch

                # Determine phase: adaptive pre-training
                if pretrain_converged:
                    is_pretrain = global_epoch < cn_pretrain_done_epoch
                else:
                    is_pretrain = global_epoch < self.cn_pretrain_max_epochs
                    if global_epoch >= self.cn_pretrain_min_epochs:
                        if self._check_pretrain_convergence(nll_history):
                            pretrain_converged = True
                            cn_pretrain_done_epoch = global_epoch
                            self.unet.history['_cn_pretrain_converged'] = True
                            self.unet.history['_cn_pretrain_done_epoch'] = global_epoch

                            # Set NLL shift: C = 2 * |final NLL|
                            final_nll = nll_history[-1]
                            self.nll_shift_C = 2.0 * abs(final_nll)
                            self.unet.history['_nll_shift_C'] = self.nll_shift_C

                            is_pretrain = False
                            print(f"\n*** Parallel phase ENDED — CN converged at epoch "
                                  f"{global_epoch} ***")
                            print(f"    NLL shift C = 2 * |{final_nll:.4f}| = "
                                  f"{self.nll_shift_C:.4f}")
                            print(f"    NLL history (last {self.cn_convergence_window + 1}): "
                                  f"{[f'{x:.4f}' for x in nll_history[-(self.cn_convergence_window + 1):]]}")

                phase_str = "PARALLEL" if is_pretrain else "JOINT"
                self.cn.apply_corrections = not is_pretrain

                # If transitioning to joint and C not yet set
                # (max-epoch fallback without convergence detection)
                if not is_pretrain and self.nll_shift_C is None:
                    if len(nll_history) > 0:
                        final_nll = nll_history[-1]
                    else:
                        # No history available — use current diagnostic
                        final_nll = self.cn.get_diagnostics().get('mean_nll', -5.0)
                    self.nll_shift_C = 2.0 * abs(final_nll)
                    self.unet.history['_nll_shift_C'] = self.nll_shift_C
                    print(f"  NLL shift C set at parallel→joint transition: "
                          f"C = 2 * |{final_nll:.4f}| = {self.nll_shift_C:.4f}")

                acc = {k: 0.0 for k in [
                    'l_total', 'l_mse', 'l_pearson', 'l_cn_own',
                    'l_cold', 'l_subgroup', 'hard_cold_train',
                    'l_weighted_mse', 'l_weighted_flow', 'l_cn_shifted']}
                n_batches = 0

                if is_pretrain:
                    # ==================================================
                    # PARALLEL PHASE: CN and UNet train independently.
                    # CN: NLL on ground temperature distribution.
                    # UNet: MSE against raw (cloud-contaminated) targets.
                    # No gradient flow between them.
                    # ==================================================
                    if self.unet.architecture == 'flow_matching':
                        self.unet.flow_model.train()
                    else:
                        self.unet.encoder.train()
                        self.unet.decoder.train()
                    self.cn.train()
                    epoch_nll_sum = 0.0
                    epoch_nll_count = 0

                    for batch in train_loader:
                        inputs, targets, _ = batch
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        if self.unet.augment:
                            inputs, targets = augment_batch(
                                inputs, targets,
                                slope_dir_channel=self.unet.slope_direction_channel)

                        # --- CN step (independent) ---
                        cn_optimizer.zero_grad()
                        with autocast():
                            _ = self.cn(inputs, targets, self._get_era5(inputs))
                            l_cn_own = self.cn.get_cn_own_loss()
                        scaler.scale(l_cn_own).backward()
                        scaler.unscale_(cn_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.cn.parameters(), max_norm=1.0)
                        scaler.step(cn_optimizer)

                        # --- UNet step (independent, vs raw targets, MSE only) ---
                        # No Pearson here: raw targets contain cloud contamination,
                        # and Pearson would reinforce contaminated spatial patterns.
                        # Pearson is introduced at joint phase with CN-corrected targets.
                        unet_optimizer.zero_grad()
                        with autocast():
                            pred = self._unet_forward(inputs)
                            l_mse = self.unet.loss_fn(pred, targets)
                        scaler.scale(l_mse).backward()
                        scaler.unscale_(unet_optimizer)
                        torch.nn.utils.clip_grad_norm_(unet_params, max_norm=1.0)
                        scaler.step(unet_optimizer)

                        # Single update per iteration (after all step() calls)
                        scaler.update()

                        acc['l_cn_own'] += l_cn_own.item()
                        acc['l_mse'] += l_mse.item()
                        n_batches += 1

                        # Track NLL for convergence detection
                        cn_diag = self.cn.get_diagnostics()
                        epoch_nll_sum += cn_diag.get('mean_nll', 0.0)
                        epoch_nll_count += 1

                    train_loss = acc['l_mse'] / max(n_batches, 1)
                    epoch_mean_nll = epoch_nll_sum / max(epoch_nll_count, 1)
                    nll_history.append(epoch_mean_nll)
                    self.unet.history['_nll_history'] = nll_history

                else:
                    # ==================================================
                    # JOINT: Both networks, learned loss balancing.
                    # ==================================================
                    if self.unet.architecture == 'flow_matching':
                        self.unet.flow_model.train()
                    else:
                        self.unet.encoder.train()
                        self.unet.decoder.train()
                    self.cn.train()

                    for batch in train_loader:
                        inputs, targets, _ = batch
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        unet_optimizer.zero_grad()
                        cn_optimizer.zero_grad()
                        lst_raw_norm = targets
                        era5_norm = self._get_era5(inputs)

                        if self.unet.augment:
                            inputs, lst_raw_norm = augment_batch(
                                inputs, lst_raw_norm,
                                slope_dir_channel=self.unet.slope_direction_channel)
                            era5_norm = self._get_era5(inputs)

                        with autocast():
                            lst_cn = self.cn(inputs, lst_raw_norm, era5_norm)
                            pred = self._unet_forward(inputs)

                            # Raw losses
                            l_mse = self.unet.loss_fn(pred, lst_cn)
                            l_cn_own = self.cn.get_cn_own_loss()

                            # Learned weighting (homoscedastic uncertainty)
                            # Shift flow loss to be positive for Kendall stability.
                            # C is a fixed constant (set at pre-train end),
                            # so dC/d(params) = 0 and CN gradients are unaffected.
                            l_cn_shifted = torch.clamp(
                                l_cn_own + self.nll_shift_C, min=SHIFT_FLOOR)

                            precision_mse = torch.exp(-self.loss_weights.log_var_mse)
                            precision_flow = torch.exp(-self.loss_weights.log_var_flow)
                            l_weighted_mse = precision_mse * l_mse + self.loss_weights.log_var_mse
                            l_weighted_flow = precision_flow * l_cn_shifted + self.loss_weights.log_var_flow

                            # Other losses (not reweighted)
                            pearson_corr = self.unet.pearson_corr_torch(pred, lst_cn)
                            l_pearson = self.unet.lambda_pearson * (
                                1 - torch.mean(pearson_corr))
                            l_cold = self.lambda_cold * self._soft_cold_pixel_rate(
                                pred, era5_norm)

                            if self.lambda_subgroup > 0:
                                l_subgroup, _ = compute_subgroup_robustness_loss(
                                    pred, lst_cn, inputs)
                                l_subgroup = self.lambda_subgroup * l_subgroup
                            else:
                                l_subgroup = torch.tensor(0.0, device=device)

                            l_total = (l_weighted_mse + l_weighted_flow
                                       + l_pearson + l_cold + l_subgroup)

                        scaler.scale(l_total).backward()
                        scaler.unscale_(unet_optimizer)
                        scaler.unscale_(cn_optimizer)
                        torch.nn.utils.clip_grad_norm_(unet_params, max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.cn.parameters()) + list(self.loss_weights.parameters()),
                            max_norm=1.0)
                        scaler.step(unet_optimizer)
                        scaler.step(cn_optimizer)
                        scaler.update()

                        acc['l_total'] += l_total.item()
                        acc['l_mse'] += l_mse.item()
                        acc['l_pearson'] += l_pearson.item()
                        acc['l_cn_own'] += l_cn_own.item()
                        acc['l_cold'] += l_cold.item()
                        acc['l_subgroup'] += (
                            l_subgroup.item() if torch.is_tensor(l_subgroup)
                            else l_subgroup)
                        acc['l_weighted_mse'] += l_weighted_mse.item()
                        acc['l_weighted_flow'] += l_weighted_flow.item()
                        acc['l_cn_shifted'] += l_cn_shifted.item()
                        acc['hard_cold_train'] += self._hard_cold_pct(pred, era5_norm)
                        n_batches += 1

                    train_loss = acc['l_mse'] / max(n_batches, 1)

                if global_epoch < T_max:
                    unet_scheduler.step()
                    cn_scheduler.step()

                # ---- TEST / LOGGING ----
                if epoch % self.unet.test_interval == 0:
                    cn_diag = self.cn.get_diagnostics()
                    avg = lambda k: acc[k] / max(n_batches, 1)
                    lr_cn = cn_optimizer.param_groups[0]['lr']

                    if is_pretrain:
                        # Quick UNet test eval during parallel phase
                        if self.unet.architecture == 'flow_matching':
                            self.unet.flow_model.eval()
                        else:
                            self.unet.encoder.eval()
                            self.unet.decoder.eval()
                        test_mse_sum = 0.0
                        n_test = 0
                        with torch.no_grad():
                            for batch in test_loader:
                                inputs_t, targets_t, _ = batch
                                inputs_t = inputs_t.to(device, non_blocking=True)
                                targets_t = targets_t.to(device, non_blocking=True)
                                pred_t = self._unet_forward(inputs_t)
                                test_mse_sum += self.unet.loss_fn(
                                    pred_t, targets_t).item()
                                n_test += 1
                        test_mse = test_mse_sum / max(n_test, 1)

                        print(f"\nepoch: {global_epoch}  [{phase_str}]")
                        print(f"  l_cn_own:   {avg('l_cn_own'):.6f}  "
                              f"mean_nll={nll_history[-1]:.4f}")
                        print(f"  l_mse:      train={avg('l_mse'):.6f}  "
                              f"test={test_mse:.6f}")
                        print(f"  CN diag:    " + "  ".join(
                            f"{k}={v:.4f}" if isinstance(v, float)
                            else f"{k}={v}"
                            for k, v in cn_diag.items()))
                        lr_unet = unet_optimizer.param_groups[0]['lr']
                        print(f"  lr:         unet={lr_unet:.2e}  cn={lr_cn:.2e}")

                        h = self.unet.history
                        h.setdefault('l_cn_own', []).append(float(avg('l_cn_own')))
                        h.setdefault('train_loss', []).append(float(avg('l_mse')))
                        h.setdefault('test_loss', []).append(float(test_mse))
                        h['nr_epochs'] = global_epoch + 1
                        self.unet.save(model_folder)
                        self._save_joint_state(
                            os.path.join(model_folder, 'joint_state'),
                            unet_optimizer, cn_optimizer,
                            unet_scheduler, cn_scheduler)

                    else:
                        # Joint: full eval
                        if self.unet.architecture == 'flow_matching':
                            self.unet.flow_model.eval()
                        else:
                            self.unet.encoder.eval()
                            self.unet.decoder.eval()
                        self.cn.eval()

                        test_loss_sum = test_pearson_sum = hard_cold_test_sum = 0.0
                        subgroup_info_test = {}
                        n_test = 0

                        with torch.no_grad():
                            for batch in test_loader:
                                inputs_t, targets_t, _ = batch
                                inputs_t = inputs_t.to(device, non_blocking=True)
                                targets_t = targets_t.to(device, non_blocking=True)
                                era5_t = self._get_era5(inputs_t)
                                pred_t = self._unet_forward(inputs_t)
                                # Test MSE vs RAW LST
                                test_loss_sum += self.unet.loss_fn(
                                    pred_t, targets_t).item()
                                pearson_t = self.unet.pearson_corr_torch(
                                    pred_t, targets_t)
                                test_pearson_sum += (
                                    1 - torch.mean(pearson_t)).item()
                                hard_cold_test_sum += self._hard_cold_pct(
                                    pred_t, era5_t)
                                _, sg_info = compute_subgroup_robustness_loss(
                                    pred_t, targets_t, inputs_t)
                                if not subgroup_info_test:
                                    subgroup_info_test = sg_info
                                n_test += 1

                        test_loss = test_loss_sum / max(n_test, 1)
                        test_pearson = test_pearson_sum / max(n_test, 1)
                        hard_cold_test = hard_cold_test_sum / max(n_test, 1)
                        hard_cold_train = acc['hard_cold_train'] / max(n_batches, 1)

                        if _ema_test is None:
                            _ema_test = test_loss
                        else:
                            _ema_test = EMA_ALPHA * test_loss + (
                                1 - EMA_ALPHA) * _ema_test

                        ratio = (test_loss / train_loss
                                 if train_loss > 0 else float('inf'))
                        if _ema_ratio is None:
                            _ema_ratio = ratio
                        else:
                            _ema_ratio = EMA_ALPHA * ratio + (
                                1 - EMA_ALPHA) * _ema_ratio

                        lr_unet = unet_optimizer.param_groups[0]['lr']
                        w_mse = torch.exp(-self.loss_weights.log_var_mse).item()
                        w_flow = torch.exp(-self.loss_weights.log_var_flow).item()

                        print(f"\nepoch: {global_epoch}  [{phase_str}]")
                        print(f"  losses:     l_mse={avg('l_mse'):.6f}  "
                              f"l_pearson={avg('l_pearson'):.6f}  "
                              f"l_cn_own={avg('l_cn_own'):.6f}  "
                              f"l_cold={avg('l_cold'):.6f}  "
                              f"l_subgroup={avg('l_subgroup'):.6f}")
                        print(f"  weighted:   w_mse*L_mse={avg('l_weighted_mse'):.6f}  "
                              f"w_flow*L_shifted={avg('l_weighted_flow'):.6f}  "
                              f"w_mse={w_mse:.4f}  w_flow={w_flow:.6f}  "
                              f"s_mse={self.loss_weights.log_var_mse.item():.3f}  "
                              f"s_flow={self.loss_weights.log_var_flow.item():.3f}")
                        print(f"  shift:      l_cn_shifted={avg('l_cn_shifted'):.6f}  "
                              f"C={self.nll_shift_C:.4f}  "
                              f"floor={SHIFT_FLOOR}")
                        print(f"  mse:        train={train_loss:.6f}  "
                              f"test={test_loss:.6f}  "
                              f"ema_test={_ema_test:.6f}  "
                              f"ema_ratio={_ema_ratio:.3f}  "
                              f"test_pearson={test_pearson:.4f}")
                        print(f"  cold%:      train={hard_cold_train:.6f}%  "
                              f"test={hard_cold_test:.6f}%")
                        print(f"  CN diag:    " + "  ".join(
                            f"{k}={v:.4f}" if isinstance(v, float)
                            else f"{k}={v}"
                            for k, v in cn_diag.items()))
                        if subgroup_info_test:
                            print(f"  subgroup:   mse_var="
                                  f"{subgroup_info_test.get('mse_var_across_bins', 0):.6f}  "
                                  f"n_groups="
                                  f"{subgroup_info_test.get('n_subgroups', 0)}")
                        print(f"  lr:         unet={lr_unet:.2e}  cn={lr_cn:.2e}")

                        h = self.unet.history
                        h.setdefault('train_loss', []).append(float(train_loss))
                        h.setdefault('test_loss', []).append(float(test_loss))
                        h.setdefault('test_pearson_loss', []).append(
                            float(test_pearson))
                        h.setdefault('l_cn_own', []).append(
                            float(avg('l_cn_own')))
                        h.setdefault('l_cold', []).append(float(avg('l_cold')))
                        h.setdefault('l_subgroup', []).append(
                            float(avg('l_subgroup')))
                        h.setdefault('hard_cold_test_pct', []).append(
                            float(hard_cold_test))
                        h.setdefault('hard_cold_train_pct', []).append(
                            float(hard_cold_train))
                        h.setdefault('cn_pct_active', []).append(
                            float(cn_diag.get('pct_active', 0)))
                        h.setdefault('log_var_mse', []).append(
                            float(self.loss_weights.log_var_mse.item()))
                        h.setdefault('log_var_flow', []).append(
                            float(self.loss_weights.log_var_flow.item()))
                        h['_ema_test_mse'] = float(_ema_test)
                        h['_ema_ratio'] = float(_ema_ratio)
                        h['nr_epochs'] = global_epoch + 1

                        if _ema_test < _best_ema_test:
                            _best_ema_test = _ema_test
                            h['_best_ema_test_mse'] = float(_best_ema_test)
                            ckpt = os.path.join(
                                model_folder, 'checkpoint_best_test_mse')
                            print(f"  ★ New best ema_test "
                                  f"{_ema_test:.6f} → {ckpt}")
                            self._save_checkpoint(
                                ckpt, unet_optimizer, cn_optimizer,
                                unet_scheduler, cn_scheduler, global_epoch)

                        if (global_epoch >= RATIO_WARMUP_EPOCHS and
                                _ema_ratio < _best_ema_ratio):
                            _best_ema_ratio = _ema_ratio
                            h['_best_train_test_ratio'] = float(_best_ema_ratio)
                            ckpt = os.path.join(
                                model_folder, 'checkpoint_best_ratio')
                            print(f"  ★ New best ema_ratio "
                                  f"{_ema_ratio:.3f} → {ckpt}")
                            self._save_checkpoint(
                                ckpt, unet_optimizer, cn_optimizer,
                                unet_scheduler, cn_scheduler, global_epoch)

                        self.unet.history['nr_epochs'] = global_epoch + 1
                        self.unet.save(model_folder)
                        self._save_joint_state(
                            os.path.join(model_folder, 'joint_state'),
                            unet_optimizer, cn_optimizer,
                            unet_scheduler, cn_scheduler)

                # Periodic checkpoint
                if (checkpoint_interval and
                        (global_epoch + 1) % checkpoint_interval == 0):
                    ckpt = os.path.join(
                        model_folder, f'checkpoint_epoch_{global_epoch + 1}')
                    print(f"Saving checkpoint to {ckpt}...")
                    self._save_checkpoint(
                        ckpt, unet_optimizer, cn_optimizer,
                        unet_scheduler, cn_scheduler, global_epoch)

                if self.unet._sigterm_received:
                    self.unet.history['nr_epochs'] = global_epoch + 1
                    self.unet.save(model_folder)
                    self._save_joint_state(
                        os.path.join(model_folder, 'joint_state'),
                        unet_optimizer, cn_optimizer,
                        unet_scheduler, cn_scheduler)
                    break

        except KeyboardInterrupt:
            print("Interrupted — saving emergency checkpoint...")
            emergency = os.path.join(model_folder, 'checkpoint_interrupted')
            self._save_checkpoint(
                emergency, unet_optimizer, cn_optimizer,
                unet_scheduler, cn_scheduler, global_epoch)

        end = time.time()
        print(f"Elapsed: {end - start:.1f}s")

        if not self.unet._sigterm_received:
            self.unet.history['nr_epochs'] = epochs_already_done + epochs_this_job
            self.unet.save(model_folder)

    def _save_checkpoint(self, ckpt_path, unet_opt, cn_opt, unet_sch, cn_sch,
                         epoch):
        self.unet.save(ckpt_path)
        torch.save(unet_opt.state_dict(),
                   os.path.join(ckpt_path, 'optimizer.state'))
        torch.save(unet_sch.state_dict(),
                   os.path.join(ckpt_path, 'scheduler.state'))
        self._save_joint_state(
            os.path.join(ckpt_path, 'joint_state'),
            unet_opt, cn_opt, unet_sch, cn_sch)
        cn_config = {
            'cn_type': 'flow',
            'cn_pretrain_min_epochs': self.cn_pretrain_min_epochs,
            'cn_pretrain_max_epochs': self.cn_pretrain_max_epochs,
            'cn_convergence_threshold': self.cn_convergence_threshold,
            'cn_convergence_window': self.cn_convergence_window,
            'lambda_cold': self.lambda_cold,
            'cold_threshold_k': self.cold_threshold_k,
            'lambda_subgroup': self.lambda_subgroup,
            'cn_lr': self.cn_lr,
            'epoch': epoch,
            'log_var_mse': self.loss_weights.log_var_mse.item(),
            'log_var_flow': self.loss_weights.log_var_flow.item(),
            'nll_shift_C': self.nll_shift_C,
        }
        with open(os.path.join(ckpt_path, 'joint_config.json'), 'w') as f:
            json.dump(cn_config, f, indent=2)

    def _save_joint_state(self, joint_dir, unet_opt, cn_opt, unet_sch, cn_sch):
        os.makedirs(joint_dir, exist_ok=True)
        torch.save(self.cn.state_dict(),
                   os.path.join(joint_dir, 'cn.weights'))
        torch.save(unet_opt.state_dict(),
                   os.path.join(joint_dir, 'unet_optimizer.state'))
        torch.save(cn_opt.state_dict(),
                   os.path.join(joint_dir, 'cn_optimizer.state'))
        torch.save(unet_sch.state_dict(),
                   os.path.join(joint_dir, 'unet_scheduler.state'))
        torch.save(cn_sch.state_dict(),
                   os.path.join(joint_dir, 'cn_scheduler.state'))
        torch.save({
            'log_var_mse': self.loss_weights.log_var_mse.data,
            'log_var_flow': self.loss_weights.log_var_flow.data,
        }, os.path.join(joint_dir, 'log_var.pt'))


def load_joint_v3_for_continue(model_folder):
    joint_config_path = os.path.join(model_folder, 'joint_config.json')
    if not os.path.exists(joint_config_path):
        raise FileNotFoundError(f"joint_config.json not found in {model_folder}.")
    with open(joint_config_path) as f:
        jc = json.load(f)

    unet = UNET()
    unet.load(model_folder)

    joint = JointUNETv3(
        unet=unet,
        cn_pretrain_min_epochs=jc.get('cn_pretrain_min_epochs', 10),
        cn_pretrain_max_epochs=jc.get('cn_pretrain_max_epochs', 200),
        cn_convergence_threshold=jc.get('cn_convergence_threshold', 0.01),
        cn_convergence_window=jc.get('cn_convergence_window', 5),
        lambda_cold=jc.get('lambda_cold', 0.1),
        cold_threshold_k=jc.get('cold_threshold_k', 10.0),
        lambda_subgroup=jc.get('lambda_subgroup', 0.1),
        cn_lr=jc.get('cn_lr', 0.0001),
    )

    # Restore log_var values
    if 'log_var_mse' in jc:
        joint.loss_weights.log_var_mse.data = torch.tensor(jc['log_var_mse'])
    if 'log_var_flow' in jc:
        joint.loss_weights.log_var_flow.data = torch.tensor(jc['log_var_flow'])

    # Restore NLL shift constant
    if 'nll_shift_C' in jc and jc['nll_shift_C'] is not None:
        joint.nll_shift_C = jc['nll_shift_C']

    cn_weights_path = os.path.join(model_folder, 'joint_state', 'cn.weights')
    if os.path.exists(cn_weights_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        joint.cn.load_state_dict(
            torch.load(cn_weights_path, map_location=device))
        print(f"Loaded CN weights from {cn_weights_path}")
    else:
        print("Warning: CN weights not found, starting CN from scratch")

    return joint
