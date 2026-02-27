"""
Joint training wrapper: CorrectionNetwork + main UNET.

At training time:
  1. CN corrects raw Landsat LST -> lst_cn
  2. Main UNET trains to predict LST using lst_cn as target
  3. Full joint backprop — single backward pass, no gradient severing

At inference time:
  - CN is discarded entirely
  - Main UNET checkpoint used directly with apply_cae — no pipeline changes

Checkpoint format:
  - UNET saved in standard format (apply_cae compatible) via unet.save()
  - CN weights + both optimizer/scheduler states saved in joint_state/ subdir
  - Continue-training restores both

v2 changes:
  - All CN diagnostics epoch-accumulated (not single-batch snapshots)
  - Speed: AMP, DataLoader workers, non_blocking, torch.compile
  - cold% printed at :.6f precision
  - CN v2: no cold prior, uniform identity loss, local gradient feature
"""

import os
import json
import time
import signal
import torch
import torch.optim as optim
import numpy as np

from cae_tools.models.unet import UNET, augment_batch
from cae_tools.models.correction_network import (
    build_correction_network,
    soft_cold_pixel_rate,
    mmd_loss_batch,
)

# Channel index of ERA5 SKT in the input tensor (must match preprocessing order)
ERA5_CHANNEL_IDX = 3

# Static-only channels fed to CN — no monthly-derived channels
# land_cover=0, elevation=2, slope_magnitude=6, slope_direction=7
CN_STATIC_CHANNEL_INDICES = [0, 2, 6, 7]

EMA_ALPHA = 0.3
RATIO_WARMUP_EPOCHS = 50


class JointUNET:
    """
    Wraps a UNET and a CorrectionNetwork for joint training.
    The UNET is fully standard and plug-out-able for inference via apply_cae.
    """

    def __init__(
        self,
        unet,
        cn_type='conv',
        cn_base_channels=32,
        cn_n_conv_layers=4,
        lambda_sparsity=0.01,
        lambda_cold=0.1,
        cold_threshold_k=10.0,
        cn_lr=0.0001,
        lambda_mmd=0.1,
        lambda_identity=1.0,
        max_mask_fraction=0.05,
        lambda_mask_cap=1.0,
    ):
        self.unet = unet
        self.lambda_cold = lambda_cold
        self.cold_threshold_k = cold_threshold_k
        self.cn_lr = cn_lr
        self.lambda_mmd = lambda_mmd

        self.cn = build_correction_network(
            cn_type=cn_type,
            static_channel_indices=CN_STATIC_CHANNEL_INDICES,
            base_channels=cn_base_channels,
            n_conv_layers=cn_n_conv_layers,
            lambda_sparsity=lambda_sparsity,
            lambda_identity=lambda_identity,
            max_mask_fraction=max_mask_fraction,
            lambda_mask_cap=lambda_mask_cap,
        )

    def _get_era5(self, inputs):
        return inputs[:, ERA5_CHANNEL_IDX:ERA5_CHANNEL_IDX+1, :, :]

    def _unet_forward(self, inputs):
        """Forward pass through whichever UNET architecture is active."""
        if self.unet.architecture == 'flow_matching':
            device = next(self.unet.flow_model.parameters()).device
            B = inputs.shape[0]
            t = torch.ones(B, device=device)
            x_t = torch.zeros(
                B, self.unet.output_shape[0],
                inputs.shape[2], inputs.shape[3], device=device
            )
            return self.unet.flow_model(x_t, inputs, t)
        else:
            enc_out, skips = self.unet.encoder(inputs)
            return self.unet.decoder(enc_out, skips)

    def _hard_cold_pct(self, pred_norm, era5_norm):
        """Non-differentiable cold pixel fraction for monitoring (returns percent)."""
        norm_params = self.unet.normalisation_parameters
        if isinstance(norm_params, list):
            min_out  = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out  = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
            era5_min = norm_params[0]['era5_skt']
            era5_max = norm_params[1]['era5_skt']
        else:
            min_out  = norm_params['min_output']
            max_out  = norm_params['max_output']
            era5_min = norm_params['min_inputs']['era5_skt']
            era5_max = norm_params['max_inputs']['era5_skt']

        pred_k = pred_norm.detach() * (max_out - min_out) + min_out
        era5_k = era5_norm.detach() * (era5_max - era5_min) + era5_min
        cold = (pred_k < (era5_k - self.cold_threshold_k)).float().mean().item()
        return cold * 100.0

    def train_joint(
        self,
        train_ds,
        test_ds,
        model_folder,
        nr_epochs,
        batch_size,
        checkpoint_interval=None,
        database_path=None,
        training_paths="",
        test_paths="",
    ):
        # ---- Setup UNET internals (mirrors train_from_datasets exactly) ----
        self.unet.set_input_spec(train_ds.get_input_spec())
        self.unet.set_output_spec(train_ds.get_output_spec())
        self.unet.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds.set_normalisation_parameters(self.unet.normalisation_parameters)

        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()
        self.unet.input_shape  = (input_chan, input_y, input_x)
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

        use_fc = (self.unet.bottleneck_type == 'fc')
        from cae_tools.models.standard_unet import StandardEncoder, StandardDecoder
        from cae_tools.models.flow_matching_unet import FlowMatchingUNet
        from cae_tools.models.unet import Encoder, Decoder

        if self.unet.architecture == 'flow_matching':
            if not self.unet.flow_model:
                self.unet.flow_model = FlowMatchingUNet(
                    cond_channels=input_chan, target_channels=output_chan,
                    base_channels=self.unet.base_channels,
                    dropout_rate=self.unet.dropout_rate
                )
        else:
            if not self.unet.encoder:
                if self.unet.architecture == 'standard':
                    self.unet.encoder = StandardEncoder(
                        in_channels=input_chan, base_channels=self.unet.base_channels,
                        dropout_rate=self.unet.dropout_rate
                    )
                else:
                    self.unet.encoder = Encoder(
                        self.unet.spec.get_input_layers(),
                        encoded_space_dim=self.unet.encoded_dim_size,
                        fc_size=self.unet.fc_size,
                        dropout_rate=self.unet.dropout_rate,
                        use_fc=use_fc,
                        latent_activation=self.unet.latent_activation
                    )
            if not self.unet.decoder:
                if self.unet.architecture == 'standard':
                    self.unet.decoder = StandardDecoder(
                        out_channels=output_chan, base_channels=self.unet.base_channels,
                        dropout_rate=self.unet.dropout_rate,
                        output_activation=self.unet.output_activation
                    )
                else:
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
                        output_activation=self.unet.output_activation
                    )

        # ---- Device ----
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Running on device: {device}")

        # ---- AMP ----
        use_amp = (device.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        if use_amp:
            print("AMP enabled (fp16 forward/backward with GradScaler)")

        if self.unet.architecture == 'flow_matching':
            self.unet.flow_model.to(device)
        else:
            self.unet.encoder.to(device)
            self.unet.decoder.to(device)
        self.cn = self.cn.to(device)

        # ---- torch.compile ----
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                if self.unet.architecture != 'flow_matching':
                    self.unet.encoder = torch.compile(self.unet.encoder)
                    self.unet.decoder = torch.compile(self.unet.decoder)
                else:
                    self.unet.flow_model = torch.compile(self.unet.flow_model)
                self.cn = torch.compile(self.cn)
                print("torch.compile enabled for encoder, decoder, CN")
            except Exception as e:
                print(f"torch.compile failed ({e}), continuing without it")

        os.makedirs(model_folder, exist_ok=True)

        # ---- DataLoaders ----
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True
        )

        # ---- Optimizers ----
        if self.unet.architecture == 'flow_matching':
            unet_params = list(self.unet.flow_model.parameters())
        else:
            unet_params = (list(self.unet.encoder.parameters()) +
                           list(self.unet.decoder.parameters()))

        unet_optimizer = torch.optim.AdamW(
            unet_params, lr=self.unet.lr, weight_decay=self.unet.weight_decay
        )
        cn_optimizer = torch.optim.AdamW(
            self.cn.parameters(), lr=self.cn_lr, weight_decay=self.unet.weight_decay
        )

        # ---- Schedulers ----
        epochs_already_done = self.unet.history.get('nr_epochs', 0)
        if 'total_nr_epochs' not in self.unet.history:
            self.unet.history['total_nr_epochs'] = nr_epochs
        T_max = self.unet.history['total_nr_epochs']
        epochs_this_job = nr_epochs

        print(f"Scheduler: T_max={T_max} (total), already done={epochs_already_done}, "
              f"this job={epochs_this_job}")

        unet_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            unet_optimizer, T_max=T_max, eta_min=1e-5
        )
        cn_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            cn_optimizer, T_max=T_max, eta_min=1e-6
        )
        self.unet._scheduler = unet_scheduler

        # ---- Restore optimizer/scheduler states if continuing ----
        if getattr(self.unet, '_saved_optimizer_state_path', None):
            print(f"Restoring UNET optimizer from {self.unet._saved_optimizer_state_path}")
            unet_optimizer.load_state_dict(
                torch.load(self.unet._saved_optimizer_state_path, map_location=device)
            )
            self.unet._saved_optimizer_state_path = None

        if getattr(self.unet, '_saved_scheduler_state_path', None):
            print(f"Restoring UNET scheduler from {self.unet._saved_scheduler_state_path}")
            unet_scheduler.load_state_dict(
                torch.load(self.unet._saved_scheduler_state_path, map_location=device)
            )
            self.unet._saved_scheduler_state_path = None

        joint_state_dir = os.path.join(model_folder, 'joint_state')
        cn_opt_path = os.path.join(joint_state_dir, 'cn_optimizer.state')
        cn_sch_path = os.path.join(joint_state_dir, 'cn_scheduler.state')
        if os.path.exists(cn_opt_path):
            print(f"Restoring CN optimizer from {cn_opt_path}")
            cn_optimizer.load_state_dict(
                torch.load(cn_opt_path, map_location=device)
            )
        if os.path.exists(cn_sch_path):
            print(f"Restoring CN scheduler from {cn_sch_path}")
            cn_scheduler.load_state_dict(
                torch.load(cn_sch_path, map_location=device)
            )

        # ---- EMA tracking ----
        _ema_test       = self.unet.history.get('_ema_test_mse', None)
        _best_ema_test  = self.unet.history.get('_best_ema_test_mse', float('inf'))
        _ema_ratio      = self.unet.history.get('_ema_ratio', None)
        _best_ema_ratio = self.unet.history.get('_best_train_test_ratio', float('inf'))

        # ---- Pre-extract norm constants ----
        _np = self.unet.normalisation_parameters
        if isinstance(_np, list):
            _min_out  = _np[2].get('ST_slices', list(_np[2].values())[0])
            _max_out  = _np[3].get('ST_slices', list(_np[3].values())[0])
            _era5_min = _np[0]['era5_skt']
            _era5_max = _np[1]['era5_skt']
        else:
            _min_out  = _np['min_output']
            _max_out  = _np['max_output']
            _era5_min = _np['min_inputs']['era5_skt']
            _era5_max = _np['max_inputs']['era5_skt']
        _out_range  = _max_out - _min_out
        _era5_range = _era5_max - _era5_min

        # ---- SIGTERM handler ----
        self.unet._sigterm_received = False
        def _handle_sigterm(signum, frame):
            print("\n[SIGTERM received] finishing current epoch then saving...")
            self.unet._sigterm_received = True
        signal.signal(signal.SIGTERM, _handle_sigterm)

        self.unet.loss_fn = torch.nn.MSELoss()
        start = time.time()

        # ---- Training loop ----
        try:
            for epoch in range(epochs_this_job):
                global_epoch = epochs_already_done + epoch

                # ============================================================
                # TRAIN PASS
                # ============================================================
                if self.unet.architecture == 'flow_matching':
                    self.unet.flow_model.train()
                else:
                    self.unet.encoder.train()
                    self.unet.decoder.train()
                self.cn.train()

                train_loss_sum  = 0.0
                l_main_sum = l_cn_sum = l_reg_sum = l_cold_sum = l_pearson_sum = l_mmd_sum = l_identity_sum = l_mask_cap_sum = 0.0
                n_batches = 0

                # Epoch-accumulated CN diagnostics (all stay on GPU until print)
                total_raw_cold     = torch.tensor(0, device=device)
                total_lifted       = torch.tensor(0, device=device)
                total_correction_k = torch.tensor(0.0, device=device)
                total_abs_corr_k   = torch.tensor(0.0, device=device)
                total_max_corr_k   = torch.tensor(0.0, device=device)
                total_active       = torch.tensor(0, device=device)
                total_pixels       = torch.tensor(0, device=device)

                last_lst_raw = last_lst_cn = last_era5 = last_pred = None

                for batch in train_loader:
                    inputs, targets, _ = batch
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    unet_optimizer.zero_grad(set_to_none=True)
                    cn_optimizer.zero_grad(set_to_none=True)

                    lst_raw_norm = targets
                    era5_norm    = self._get_era5(inputs)

                    if self.unet.augment:
                        inputs, lst_raw_norm = augment_batch(
                            inputs, lst_raw_norm,
                            slope_dir_channel=self.unet.slope_direction_channel
                        )
                        era5_norm = self._get_era5(inputs)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        lst_cn = self.cn(inputs, lst_raw_norm, era5_norm)
                        pred = self._unet_forward(inputs)

                        l_mse  = self.unet.loss_fn(pred, lst_cn)
                        pearson_corr = self.unet.pearson_corr_torch(pred, lst_cn)
                        l_pearson = self.unet.lambda_pearson * (1 - torch.mean(pearson_corr))
                        l_main = l_mse + l_pearson
                        l_reg  = self.cn.get_regularisation_loss()
                        l_cold = self.lambda_cold * soft_cold_pixel_rate(
                            pred, era5_norm, self.unet.normalisation_parameters,
                            cold_threshold_k=self.cold_threshold_k
                        )
                        mask_m = self.cn.get_last_mask()
                        l_mmd = mmd_loss_batch(lst_cn, mask_m, self.lambda_mmd)
                        l_identity = self.cn.get_identity_loss()
                        l_mask_cap = self.cn.get_mask_cap_loss()
                        l_cn_total = l_reg + l_cold + l_mmd + l_identity + l_mask_cap
                        l_total    = l_main + l_cn_total

                    scaler.scale(l_total).backward()
                    scaler.unscale_(unet_optimizer)
                    scaler.unscale_(cn_optimizer)
                    torch.nn.utils.clip_grad_norm_(unet_params, max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.cn.parameters(), max_norm=1.0)
                    scaler.step(unet_optimizer)
                    scaler.step(cn_optimizer)
                    scaler.update()

                    train_loss_sum      += l_main.item()
                    l_main_sum          += l_main.item()
                    l_pearson_sum       += l_pearson.item()
                    l_mmd_sum           += l_mmd.item()
                    l_identity_sum      += l_identity.item()
                    l_mask_cap_sum      += l_mask_cap.item()
                    l_cn_sum            += l_cn_total.item()
                    l_reg_sum           += l_reg.item()
                    l_cold_sum          += l_cold.item()
                    n_batches           += 1

                    # Accumulate CN diagnostics on GPU (no .item() per batch)
                    with torch.no_grad():
                        _lst_cn_d = lst_cn.detach()
                        _corr_k = (_lst_cn_d - lst_raw_norm) * _out_range
                        _abs_ck = _corr_k.abs()
                        total_correction_k += _corr_k.sum()
                        total_abs_corr_k   += _abs_ck.sum()
                        total_max_corr_k    = torch.max(total_max_corr_k, _abs_ck.max())
                        total_active       += (_abs_ck > (0.01 * _out_range)).sum()
                        total_pixels       += _corr_k.numel()

                        _lst_raw_k = lst_raw_norm * _out_range + _min_out
                        _era5_k    = era5_norm * _era5_range + _era5_min
                        _rc = (_lst_raw_k < (_era5_k - self.cold_threshold_k))
                        total_raw_cold += _rc.sum()
                        total_lifted   += (_rc & ((_lst_cn_d * _out_range + _min_out) >= (_era5_k - self.cold_threshold_k))).sum()

                    last_lst_raw = lst_raw_norm.detach()
                    last_lst_cn  = _lst_cn_d
                    last_era5    = era5_norm.detach()
                    last_pred    = pred.detach()

                train_loss = train_loss_sum / max(n_batches, 1)

                if global_epoch < T_max:
                    unet_scheduler.step()
                    cn_scheduler.step()

                # ============================================================
                # TEST PASS + MONITORING
                # ============================================================
                if epoch % self.unet.test_interval == 0:
                    if self.unet.architecture == 'flow_matching':
                        self.unet.flow_model.eval()
                    else:
                        self.unet.encoder.eval()
                        self.unet.decoder.eval()
                    self.cn.eval()

                    test_loss_sum = 0.0
                    test_pearson_sum = 0.0
                    hard_cold_test_sum = 0.0
                    n_test = 0

                    with torch.no_grad():
                        for batch in test_loader:
                            inputs, targets, _ = batch
                            inputs  = inputs.to(device, non_blocking=True)
                            targets = targets.to(device, non_blocking=True)
                            era5_t  = self._get_era5(inputs)
                            with torch.amp.autocast('cuda', enabled=use_amp):
                                pred = self._unet_forward(inputs)
                            pred_f32 = pred.float()
                            test_loss_sum      += self.unet.loss_fn(pred_f32, targets).item()
                            pearson_corr_t      = self.unet.pearson_corr_torch(pred_f32, targets)
                            test_pearson_sum   += (1 - torch.mean(pearson_corr_t)).item()
                            hard_cold_test_sum += self._hard_cold_pct(pred_f32, era5_t)
                            n_test += 1

                    test_loss = test_loss_sum / max(n_test, 1)
                    test_pearson_loss = test_pearson_sum / max(n_test, 1)
                    hard_cold_test_pct  = hard_cold_test_sum  / max(n_test, 1)
                    hard_cold_train_pct = self._hard_cold_pct(last_pred, last_era5) if last_pred is not None else 0.0

                    # EMA
                    if _ema_test is None:
                        _ema_test = test_loss
                    else:
                        _ema_test = EMA_ALPHA * test_loss + (1 - EMA_ALPHA) * _ema_test

                    ratio = test_loss / train_loss if train_loss > 0 else float('inf')
                    if _ema_ratio is None:
                        _ema_ratio = ratio
                    else:
                        _ema_ratio = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * _ema_ratio

                    # Epoch-accumulated CN diagnostics (single GPU->CPU sync)
                    _trc        = total_raw_cold.item()
                    _tli        = total_lifted.item()
                    _tpx        = max(total_pixels.item(), 1)
                    _mean_dk    = total_correction_k.item() / _tpx
                    _mean_abs_k = total_abs_corr_k.item() / _tpx
                    _max_abs_k  = total_max_corr_k.item()
                    _pct_active = total_active.item() / _tpx * 100.0
                    _lifted_pct = (_tli / _trc * 100.0) if _trc > 0 else 100.0

                    avg = lambda s: s / max(n_batches, 1)
                    lr_unet = unet_optimizer.param_groups[0]['lr']
                    lr_cn   = cn_optimizer.param_groups[0]['lr']

                    # ---- Print block ----
                    print(f"\nepoch: {global_epoch}")
                    print(f"  losses:     l_main={avg(l_main_sum):.6f}  "
                          f"l_mse={avg(l_main_sum)-avg(l_pearson_sum):.6f}  "
                          f"l_pearson={avg(l_pearson_sum):.6f}  "
                          f"l_cn={avg(l_cn_sum):.6f}  "
                          f"l_reg={avg(l_reg_sum):.6f}  "
                          f"l_cold={avg(l_cold_sum):.6f}  "
                          f"l_mmd={avg(l_mmd_sum):.6f}  "
                          f"l_identity={avg(l_identity_sum):.6f}  "
                          f"l_mask_cap={avg(l_mask_cap_sum):.6f}")
                    print(f"  mse:        train={train_loss:.6f}  test={test_loss:.6f}  "
                          f"ema_test={_ema_test:.6f}  ema_ratio={_ema_ratio:.3f}  "
                          f"test_pearson={test_pearson_loss:.4f}")
                    print(f"  cold%:      train={hard_cold_train_pct:.6f}%  "
                          f"test={hard_cold_test_pct:.6f}%")
                    print(f"  CN target:  mean_delta={_mean_dk:+.4f}K  (epoch-avg)")
                    print(f"  CN corr:    mean={_mean_abs_k:.4f}K  "
                          f"max={_max_abs_k:.3f}K  "
                          f"pct_active={_pct_active:.2f}%  (epoch-avg)")
                    print(f"  CN cold:    raw_cold_px={_trc:.0f}  "
                          f"lifted={_lifted_pct:.1f}%  (epoch-total)")
                    print(f"  lr:         unet={lr_unet:.2e}  cn={lr_cn:.2e}")

                    # Update history
                    h = self.unet.history
                    h.setdefault('train_loss', []).append(float(train_loss))
                    h.setdefault('test_loss',       []).append(float(test_loss))
                    h.setdefault('test_pearson_loss',[]).append(float(test_pearson_loss))
                    h.setdefault('l_main',     []).append(float(avg(l_main_sum)))
                    h.setdefault('l_pearson',  []).append(float(avg(l_pearson_sum)))
                    h.setdefault('l_mmd',      []).append(float(avg(l_mmd_sum)))
                    h.setdefault('l_identity', []).append(float(avg(l_identity_sum)))
                    h.setdefault('l_mask_cap', []).append(float(avg(l_mask_cap_sum)))
                    h.setdefault('l_cn',       []).append(float(avg(l_cn_sum)))
                    h.setdefault('l_reg',       []).append(float(avg(l_reg_sum)))
                    h.setdefault('l_cold',      []).append(float(avg(l_cold_sum)))
                    h.setdefault('hard_cold_test_pct',  []).append(float(hard_cold_test_pct))
                    h.setdefault('hard_cold_train_pct', []).append(float(hard_cold_train_pct))
                    h.setdefault('cn_mean_correction_k',[]).append(float(_mean_abs_k))
                    h.setdefault('cn_pct_active',       []).append(float(_pct_active))
                    h.setdefault('cn_cold_removed_pct', []).append(float(_lifted_pct))
                    h['_ema_test_mse']          = float(_ema_test)
                    h['_ema_ratio']             = float(_ema_ratio)
                    h['nr_epochs']              = global_epoch + 1

                    # Best checkpoints
                    if _ema_test < _best_ema_test:
                        _best_ema_test = _ema_test
                        h['_best_ema_test_mse'] = float(_best_ema_test)
                        ckpt = os.path.join(model_folder, 'checkpoint_best_test_mse')
                        print(f"  ★ New best ema_test {_ema_test:.6f} → {ckpt}")
                        self._save_checkpoint(ckpt, unet_optimizer, cn_optimizer,
                                              unet_scheduler, cn_scheduler, global_epoch)

                    if (global_epoch >= RATIO_WARMUP_EPOCHS and
                            _ema_ratio > _best_ema_ratio):
                        _best_ema_ratio = _ema_ratio
                        h['_best_train_test_ratio'] = float(_best_ema_ratio)
                        ckpt = os.path.join(model_folder, 'checkpoint_best_ratio')
                        print(f"  ★ New best ema_ratio {_ema_ratio:.3f} → {ckpt}")
                        self._save_checkpoint(ckpt, unet_optimizer, cn_optimizer,
                                              unet_scheduler, cn_scheduler, global_epoch)

                    # Save root state every test_interval
                    self.unet.history['nr_epochs'] = global_epoch + 1
                    self.unet.save(model_folder)
                    self._save_joint_state(
                        os.path.join(model_folder, 'joint_state'),
                        unet_optimizer, cn_optimizer,
                        unet_scheduler, cn_scheduler
                    )

                # Periodic checkpoint
                if checkpoint_interval and (global_epoch + 1) % checkpoint_interval == 0:
                    ckpt = os.path.join(model_folder, f'checkpoint_epoch_{global_epoch+1}')
                    print(f"Saving checkpoint to {ckpt}...")
                    self._save_checkpoint(ckpt, unet_optimizer, cn_optimizer,
                                          unet_scheduler, cn_scheduler, global_epoch)

                if self.unet._sigterm_received:
                    self.unet.history['nr_epochs'] = global_epoch + 1
                    self.unet.save(model_folder)
                    self._save_joint_state(
                        os.path.join(model_folder, 'joint_state'),
                        unet_optimizer, cn_optimizer,
                        unet_scheduler, cn_scheduler
                    )
                    break

        except KeyboardInterrupt:
            print("Interrupted — saving emergency checkpoint...")
            emergency = os.path.join(model_folder, 'checkpoint_interrupted')
            self._save_checkpoint(emergency, unet_optimizer, cn_optimizer,
                                  unet_scheduler, cn_scheduler, global_epoch)

        end = time.time()
        print(f"Elapsed: {end - start:.1f}s")

        if not self.unet._sigterm_received:
            self.unet.history['nr_epochs'] = epochs_already_done + epochs_this_job
            self.unet.save(model_folder)

    # =========================================================================
    # Save helpers
    # =========================================================================

    def _save_checkpoint(self, ckpt_path, unet_opt, cn_opt, unet_sch, cn_sch, epoch):
        """Save UNET (apply_cae compatible) + joint state."""
        self.unet.save(ckpt_path)
        torch.save(unet_opt.state_dict(),
                   os.path.join(ckpt_path, 'optimizer.state'))
        torch.save(unet_sch.state_dict(),
                   os.path.join(ckpt_path, 'scheduler.state'))
        self._save_joint_state(
            os.path.join(ckpt_path, 'joint_state'),
            unet_opt, cn_opt, unet_sch, cn_sch
        )
        cn_config = {
            'cn_type':               'conv',
            'static_channel_indices': self.cn.static_channel_indices,
            'base_channels':         self.cn.net[0].out_channels
                                     if hasattr(self.cn.net[0], 'out_channels') else 32,
            'lambda_sparsity':       self.cn.lambda_sparsity,
            'lambda_cold':           self.lambda_cold,
            'cold_threshold_k':      self.cold_threshold_k,
            'cn_lr':                 self.cn_lr,
            'epoch':                 epoch,
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


# =============================================================================
# Load for continue-training
# =============================================================================

def load_joint_unet_for_continue(model_folder):
    """Load JointUNET from model_folder for continue-training."""
    joint_config_path = os.path.join(model_folder, 'joint_config.json')
    if not os.path.exists(joint_config_path):
        raise FileNotFoundError(
            f"joint_config.json not found in {model_folder}. "
            f"Was this saved by joint training?"
        )
    with open(joint_config_path) as f:
        jc = json.load(f)

    unet = UNET()
    unet.load(model_folder)

    joint = JointUNET(
        unet=unet,
        cn_type=jc.get('cn_type', 'conv'),
        cn_base_channels=jc.get('base_channels', 32),
        lambda_sparsity=jc.get('lambda_sparsity', 0.01),
        lambda_cold=jc.get('lambda_cold', 0.1),
        cold_threshold_k=jc.get('cold_threshold_k', 10.0),
        cn_lr=jc.get('cn_lr', 0.0001),
    )

    # Load CN weights
    cn_weights_path = os.path.join(model_folder, 'joint_state', 'cn.weights')
    if os.path.exists(cn_weights_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        joint.cn.load_state_dict(
            torch.load(cn_weights_path, map_location=device)
        )
        print(f"Loaded CN weights from {cn_weights_path}")
    else:
        print("Warning: CN weights not found, starting CN from scratch")

    return joint
