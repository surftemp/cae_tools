"""
Joint UNET + Flow CN v3 Trainer — Flow-Only with Learned Loss Balancing

Architecture:
  - CN computes p(LST | inputs) and produces soft-blended targets.
  - UNet trains against soft-blended targets during joint phase.
  - No fixed threshold. Correction strength determined by flow's density.
  - No AMP. Pure fp32 to match standalone baseline.

Loss formulation:
  L_total = exp(-s_mse) * L_mse + s_mse
          + exp(-s_flow) * L_flow_shifted + s_flow
          + L_pearson + L_cold + L_subgroup
  where L_flow_shifted = max(l_cn_own + C, epsilon)
  and C = 2 * |final_pretrain_nll| (fixed constant set at transition).

Standing instruction: no VGG/perceptual loss code.
"""

import os
import sys
import time
import json
import signal
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cae_tools.models.unet import UNET, augment_batch
from cae_tools.models.correction_network_v3 import (
    FlowCorrectionNetwork,
    CorrectionNetworkBase,
)
from cae_tools.models.flow_matching_unet import (
    flow_matching_loss,
    flow_matching_sample,
)

ERA5_CHANNEL_IDX = 3
EMA_ALPHA = 0.3
RATIO_WARMUP_EPOCHS = 50
SHIFT_FLOOR = 0.5  # Floor for shifted flow loss to prevent Kendall weight divergence
DIAG_IMAGE_INTERVAL = 50  # Epochs between diagnostic images
DIAG_N_BOXES = 32  # Number of boxes to plot
MAX_PRECISION = 1000  # Cap for Kendall precision to prevent gradient explosion

def compute_spectral_loss(pred, target):
    """
    Spectral loss: penalises mismatch in spatial frequency content.

    Computes 2D FFT of both pred and target (each of shape (B, 1, H, W)),
    takes the magnitude spectrum (amplitude at each spatial frequency),
    applies log(1 + |F|) to compress dynamic range so high-frequency
    components are not swamped by low-frequency ones, then returns the
    mean squared difference between the two log-magnitude spectra.

    Args:
        pred: model prediction, tensor of shape (B, 1, H, W)
        target: training target, tensor of shape (B, 1, H, W)

    Returns:
        Scalar tensor: mean squared difference of log-magnitude spectra,
        averaged over all batch elements and all spatial frequencies.
    """
    # fft2 operates on last two dims (H, W). Output is complex-valued,
    # same shape (B, 1, H, W).
    fft_pred = torch.fft.fft2(pred)
    fft_target = torch.fft.fft2(target)

    # |F| is the magnitude (amplitude) at each spatial frequency (u, v).
    # log(1 + |F|) compresses the range: low-freq components have |F| ~ 100,
    # high-freq components have |F| ~ 0.01, so without log the loss would
    # be dominated by low frequencies which are already captured by MSE.
    log_mag_pred = torch.log1p(fft_pred.abs())
    log_mag_target = torch.log1p(fft_target.abs())

    return F.mse_loss(log_mag_pred, log_mag_target)


def compute_subgroup_robustness_loss(pred, target, inputs, n_era5_bins=None,
                                     min_samples=50):
    if n_era5_bins is None:
        n_era5_bins = pred.shape[0]  # one bin per sample in batch
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
    robustness_loss = subgroup_mses.var()
    info = {f'mse_era5_bin_{i}': m.item() for i, m in enumerate(subgroup_mses)}
    info['n_subgroups'] = len(subgroup_mses)
    info['mse_var_across_bins'] = robustness_loss.item()
    return robustness_loss, info


def _save_diagnostic_images(save_dir, global_epoch, inputs, targets, lst_cn,
                            pred, nll_map, weight_map, x_mean_map, cn_model,
                            n_boxes=DIAG_N_BOXES):
    """
    Save diagnostic images: 2x4 panel per box + density curve.

    Panels:
        (a) Raw LST target
        (b) NLL heatmap — high values = flow thinks observed LST is unlikely
        (c) Blending weight w — 0.5=clean, 0.0=fully corrected
        (d) Flow conditional mean (x_mean) — what the flow thinks LST should be
        (e) Effective target (lst_cn) — what UNet actually trains against
        (f) UNet prediction
        (g) Error (pred - raw target)
        (h) Correction (x_mean - raw target) — what the flow wants to change
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [diag] matplotlib not available, skipping diagnostic images")
        return

    os.makedirs(save_dir, exist_ok=True)
    n = min(n_boxes, inputs.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        raw = targets[i, 0].cpu().numpy()
        nll = nll_map[i, 0].cpu().numpy()
        wgt = weight_map[i, 0].cpu().numpy()
        xm = x_mean_map[i, 0].cpu().numpy()
        corrected = lst_cn[i, 0].detach().cpu().numpy()
        prediction = pred[i, 0].detach().cpu().numpy()
        error = prediction - raw
        correction = xm - raw

        # (a) Raw LST target
        im0 = axes[0, 0].pcolormesh(raw, cmap='inferno')
        axes[0, 0].set_title('Raw LST target')
        plt.colorbar(im0, ax=axes[0, 0])

        # (b) NLL heatmap
        im1 = axes[0, 1].pcolormesh(nll, cmap='hot')
        axes[0, 1].set_title(f'NLL (mean={nll.mean():.2f})')
        plt.colorbar(im1, ax=axes[0, 1])

        # (c) Blending weight w
        im2 = axes[0, 2].pcolormesh(wgt, cmap='RdYlGn', vmin=0, vmax=0.5)
        axes[0, 2].set_title(f'Weight w (0=corrected, 0.5=clean)\n'
                             f'pct<0.1: {(wgt < 0.1).mean()*100:.2f}%')
        plt.colorbar(im2, ax=axes[0, 2])

        # (d) Flow conditional mean (x_mean)
        im3 = axes[0, 3].pcolormesh(xm, cmap='inferno',
                                     vmin=raw.min(), vmax=raw.max())
        axes[0, 3].set_title('Flow x_mean (conditional mode)')
        plt.colorbar(im3, ax=axes[0, 3])

        # (e) Effective target (lst_cn)
        im4 = axes[1, 0].pcolormesh(corrected, cmap='inferno',
                                     vmin=raw.min(), vmax=raw.max())
        axes[1, 0].set_title('Effective target (lst_cn)')
        plt.colorbar(im4, ax=axes[1, 0])

        # (f) UNet prediction
        im5 = axes[1, 1].pcolormesh(prediction, cmap='inferno',
                                     vmin=raw.min(), vmax=raw.max())
        axes[1, 1].set_title('UNet prediction')
        plt.colorbar(im5, ax=axes[1, 1])

        # (g) Prediction error (pred - raw target)
        err_max = max(abs(error.min()), abs(error.max()), 0.01)
        im6 = axes[1, 2].pcolormesh(error, cmap='RdBu_r',
                                     vmin=-err_max, vmax=err_max)
        axes[1, 2].set_title(f'Error (RMSE={np.sqrt((error**2).mean()):.4f})')
        plt.colorbar(im6, ax=axes[1, 2])

        # (h) Correction applied (x_mean - raw target)
        corr_max = max(abs(correction.min()), abs(correction.max()), 0.01)
        im7 = axes[1, 3].pcolormesh(correction, cmap='RdBu_r',
                                     vmin=-corr_max, vmax=corr_max)
        axes[1, 3].set_title(f'Correction (x_mean - raw)\n'
                             f'mean={correction.mean():.4f}')
        plt.colorbar(im7, ax=axes[1, 3])

        for ax in axes.flat:
            ax.set_aspect('equal')
            ax.invert_yaxis()

        fig.suptitle(f'Epoch {global_epoch}, Box {i}', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'epoch{global_epoch:04d}_box{i:02d}.png'),
                    dpi=100)
        plt.close(fig)

    # Density curve for the highest-NLL pixel in the first box
    try:
        nll_box0 = nll_map[0, 0].cpu()
        max_idx = nll_box0.argmax()
        py = (max_idx // nll_box0.shape[1]).item()
        px = (max_idx % nll_box0.shape[1]).item()

        cn_model.eval()
        with torch.no_grad():
            lst_vals, densities = cn_model.compute_density_curve(
                inputs[0:1], py, px, n_points=200)
        cn_model.train()

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        ax2.plot(lst_vals, densities, 'b-', linewidth=1.5)
        observed = targets[0, 0, py, px].cpu().item()
        ax2.axvline(observed, color='r', linestyle='--', linewidth=1.5,
                    label=f'Observed LST={observed:.3f}')
        # Also show x_mean
        cn_model.eval()
        with torch.no_grad():
            context = cn_model._get_context(inputs[0:1])
            x_mean_val = cn_model._flow_inverse(
                torch.zeros(1, 1, 1, 1, device=inputs.device),
                context[:, :, py:py+1, px:px+1]
            ).item()
        cn_model.train()
        ax2.axvline(x_mean_val, color='g', linestyle='--', linewidth=1.5,
                    label=f'Flow mean={x_mean_val:.3f}')
        ax2.set_xlabel('LST (normalised)')
        ax2.set_ylabel('p(LST | context)')
        ax2.set_title(f'Epoch {global_epoch}, Box 0, pixel ({py},{px}), '
                      f'NLL={nll_box0[py, px]:.2f}')
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir,
                                  f'epoch{global_epoch:04d}_density_curve.png'),
                     dpi=100)
        plt.close(fig2)
    except Exception as e:
        print(f"  [diag] density curve failed: {e}")


class JointUNETv3:
    """
    Joint UNET + Flow CN v3 trainer.

    Learned loss balancing (homoscedastic uncertainty weighting):
        L = exp(-s_mse) * L_mse + s_mse
          + exp(-s_flow) * max(L_cn_own + C, eps) + s_flow
          + L_pearson + L_cold + L_subgroup
    """

    def __init__(
        self,
        unet,
        cn_base_channels=64,
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
        lambda_spectral=0.0,
        spectral_every_k_epochs=10,
        # Flow CN specific
        flow_n_coupling_layers=4,
        flow_coupling_hidden=64,
    ):
        self.unet = unet
        self.cn_lr = cn_lr

        self.cn_pretrain_min_epochs = cn_pretrain_min_epochs
        self.cn_pretrain_max_epochs = cn_pretrain_max_epochs
        self.cn_convergence_threshold = cn_convergence_threshold
        self.cn_convergence_window = cn_convergence_window

        self.lambda_cold = lambda_cold
        self.cold_threshold_k = cold_threshold_k
        self.lambda_subgroup = lambda_subgroup
        self.lambda_spectral = lambda_spectral
        self.spectral_every_k_epochs = spectral_every_k_epochs

        # Store flow params for deferred CN construction.
        # CN is built in train_joint() once we know n_input_channels from data.
        self._cn_base_channels = cn_base_channels
        self._cn_dropout_rate = cn_dropout_rate
        self._flow_n_coupling_layers = flow_n_coupling_layers
        self._flow_coupling_hidden = flow_coupling_hidden
        self.cn = None  # Built in train_joint()

        # Learned loss balancing parameters (Kendall & Gal 2018)
        self.loss_weights = nn.Module()
        self.loss_weights.log_var_mse = nn.Parameter(torch.tensor(0.0))
        self.loss_weights.log_var_flow = nn.Parameter(torch.tensor(0.0))

        self.nll_shift_C = None  # Set during training

    def _build_cn(self, n_input_channels):
        """Build the CN once we know the input channel count from data."""
        self.cn = FlowCorrectionNetwork(
            n_input_channels=n_input_channels,
            base_channels=self._cn_base_channels,
            n_coupling_layers=self._flow_n_coupling_layers,
            coupling_hidden=self._flow_coupling_hidden,
            dropout_rate=self._cn_dropout_rate,
        )

    def _get_era5(self, inputs):
        return inputs[:, ERA5_CHANNEL_IDX:ERA5_CHANNEL_IDX + 1, :, :]

    def _get_output_range_k(self):
        """Return (max_output - min_output) in Kelvin for denormalisation."""
        norm_params = self.unet.normalisation_parameters
        if isinstance(norm_params, list):
            min_out = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
        else:
            min_out = norm_params['min_output']
            max_out = norm_params['max_output']
        return max_out - min_out

    def _unet_forward(self, inputs):
        """
        Forward pass through the main model.

        For standard/legacy UNet: single encoder→decoder pass.
        For flow matching: 4-step Euler integration from noise to prediction.
            This is used for test evaluation and diagnostics (under torch.no_grad).
        """
        if self.unet.architecture == 'flow_matching':
            device = next(self.unet.flow_model.parameters()).device
            B = inputs.shape[0]
            target_shape = (B, self.unet.output_shape[0],
                            inputs.shape[2], inputs.shape[3])
            return flow_matching_sample(
                self.unet.flow_model, inputs, target_shape,
                num_steps=self.unet.flow_steps, device=device)
        else:
            enc_out, skips = self.unet.encoder(inputs)
            return self.unet.decoder(enc_out, skips)

    def _euler_sample_with_grad(self, inputs):
        """
        Euler integration with gradient tracking, for computing pattern losses
        during flow matching training.

        Runs self.unet.flow_steps (default 4) Euler steps from random noise
        z ~ N(0,1) at t=0 to predicted output at t=1. All operations retain
        gradients so that pattern losses on the final output can backpropagate
        through all 4 sequential forward passes of the flow model.

        Args:
            inputs: (B, C_in, H, W) conditioning channels

        Returns:
            x: (B, C_out, H, W) predicted output after Euler integration
        """
        device = next(self.unet.flow_model.parameters()).device
        B = inputs.shape[0]
        target_shape = (B, self.unet.output_shape[0],
                        inputs.shape[2], inputs.shape[3])

        x = torch.randn(target_shape, device=device)
        n_steps = self.unet.flow_steps
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)
            v = self.unet.flow_model(x, inputs, t)
            x = x + dt * v

        return x

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

        # ---- Build CN now that we know input_chan ----
        if self.cn is None:
            self._build_cn(input_chan)

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
        diag_dir = os.path.join(model_folder, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # Save training command for reproducibility
        cmd_path = os.path.join(model_folder, 'training_command.txt')
        with open(cmd_path, 'w') as f:
            f.write(' '.join(sys.argv))
        print(f"Training command saved to {cmd_path}")

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

        cn_optimizer = torch.optim.AdamW(
            list(self.cn.parameters()) + list(self.loss_weights.parameters()),
            lr=self.cn_lr, weight_decay=self.unet.weight_decay)

        epochs_already_done = self.unet.history.get('nr_epochs', 0)
        if 'total_nr_epochs' not in self.unet.history:
            self.unet.history['total_nr_epochs'] = nr_epochs
        T_max = self.unet.history['total_nr_epochs']
        epochs_this_job = nr_epochs

        pretrain_converged = self.unet.history.get('_cn_pretrain_converged', False)
        cn_pretrain_done_epoch = self.unet.history.get(
            '_cn_pretrain_done_epoch', self.cn_pretrain_max_epochs)
        nll_history = self.unet.history.get('_nll_history', [])

        if self.nll_shift_C is None:
            self.nll_shift_C = self.unet.history.get('_nll_shift_C', None)

        print(f"Scheduler: T_max={T_max}, already done={epochs_already_done}, "
              f"this job={epochs_this_job}")
        print(f"CN type: flow (soft blending, no threshold)")
        print(f"CN input channels: {self.cn.n_input_channels} (same as UNet)")
        print(f"Adaptive parallel phase: min={self.cn_pretrain_min_epochs}, "
              f"max={self.cn_pretrain_max_epochs}, "
              f"conv_thresh={self.cn_convergence_threshold}, "
              f"conv_window={self.cn_convergence_window}")
        print(f"Learned loss balancing: log_var_mse={self.loss_weights.log_var_mse.item():.3f}, "
              f"log_var_flow={self.loss_weights.log_var_flow.item():.3f}")
        print(f"UNet output activation: {self.unet.output_activation}")
        print(f"Precision: fp32 (no AMP)")
        print(f"Loss weights: lambda_cold={self.lambda_cold}, "
              f"lambda_subgroup={self.lambda_subgroup}, "
              f"lambda_spectral={self.lambda_spectral}, "
              f"spectral_every_k_epochs={self.spectral_every_k_epochs}")
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

        _ema_test = self.unet.history.get('_ema_test_mse', None)
        _best_test_mse = self.unet.history.get('_best_test_mse', float('inf'))
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
                epoch_start = time.time()
                global_epoch = epochs_already_done + epoch

                # Determine phase
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

                if not is_pretrain and self.nll_shift_C is None:
                    if len(nll_history) > 0:
                        final_nll = nll_history[-1]
                    else:
                        final_nll = self.cn.get_diagnostics().get('mean_nll', -5.0)
                    self.nll_shift_C = 2.0 * abs(final_nll)
                    self.unet.history['_nll_shift_C'] = self.nll_shift_C
                    print(f"  NLL shift C set at parallel->joint transition: "
                          f"C = 2 * |{final_nll:.4f}| = {self.nll_shift_C:.4f}")

                acc = {k: 0.0 for k in [
                    'l_total', 'l_mse', 'l_mae', 'l_pearson', 'l_cn_own',
                    'l_cold', 'l_subgroup', 'l_spectral', 'hard_cold_train',
                    'l_weighted_mse', 'l_weighted_flow', 'l_cn_shifted']}
                n_batches = 0

                if is_pretrain:
                    # ==================================================
                    # PARALLEL PHASE
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

                        # --- CN step ---
                        cn_optimizer.zero_grad()
                        _ = self.cn(inputs, targets)
                        l_cn_own = self.cn.get_cn_own_loss()
                        l_cn_own.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.cn.parameters(), max_norm=1.0)
                        cn_optimizer.step()

                        # --- UNet step ---
                        unet_optimizer.zero_grad()
                        if self.unet.architecture == 'flow_matching':
                            l_mse = flow_matching_loss(
                                self.unet.flow_model, inputs, targets,
                                device=device)
                        else:
                            pred = self._unet_forward(inputs)
                            l_mse = self.unet.loss_fn(pred, targets)
                        l_mse.backward()
                        torch.nn.utils.clip_grad_norm_(unet_params, max_norm=1.0)
                        unet_optimizer.step()

                        acc['l_cn_own'] += l_cn_own.item()
                        acc['l_mse'] += l_mse.item()
                        if self.unet.architecture != 'flow_matching':
                            with torch.no_grad():
                                acc['l_mae'] += (pred - targets).abs().mean().item()
                        n_batches += 1

                        cn_diag = self.cn.get_diagnostics()
                        epoch_nll_sum += cn_diag.get('mean_nll', 0.0)
                        epoch_nll_count += 1

                    train_loss = acc['l_mse'] / max(n_batches, 1)
                    epoch_mean_nll = epoch_nll_sum / max(epoch_nll_count, 1)
                    nll_history.append(epoch_mean_nll)
                    self.unet.history['_nll_history'] = nll_history

                else:
                    # ==================================================
                    # JOINT PHASE
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

                        if self.unet.augment:
                            inputs, lst_raw_norm = augment_batch(
                                inputs, lst_raw_norm,
                                slope_dir_channel=self.unet.slope_direction_channel)

                        lst_cn = self.cn(inputs, lst_raw_norm)
                        era5_norm = self._get_era5(inputs)
                        l_cn_own = self.cn.get_cn_own_loss()

                        is_fm = (self.unet.architecture == 'flow_matching')
                        is_pattern_epoch = (
                            global_epoch % self.spectral_every_k_epochs == 0)

                        # ---- Primary loss ----
                        # For standard UNet: MSE between direct prediction and
                        #   CN-corrected target. pred is a (B,1,H,W) tensor.
                        # For flow matching: CFM velocity MSE. pred is not
                        #   available (would require Euler integration).
                        if is_fm:
                            l_mse = flow_matching_loss(
                                self.unet.flow_model, inputs, lst_cn,
                                device=device)
                        else:
                            pred = self._unet_forward(inputs)
                            l_mse = self.unet.loss_fn(pred, lst_cn)

                        # ---- Kendall balancing (same for both architectures) ----
                        l_cn_shifted = torch.clamp(
                            l_cn_own + self.nll_shift_C, min=SHIFT_FLOOR)
                        
                        precision_mse = torch.clamp(
                            torch.exp(-self.loss_weights.log_var_mse),
                            max=MAX_PRECISION)
                        precision_flow = torch.clamp(
                            torch.exp(-self.loss_weights.log_var_flow),
                            
                            max=MAX_PRECISION)
                        l_weighted_mse = (precision_mse * l_mse
                                          + self.loss_weights.log_var_mse)
                        l_weighted_flow = (precision_flow * l_cn_shifted
                                           + self.loss_weights.log_var_flow)

                        # ---- Pattern losses ----
                        # Standard UNet: Pearson/cold/subgroup every batch
                        #   (pred already computed above). Spectral on K-th
                        #   epochs only.
                        # Flow matching: ALL pattern losses on K-th epochs
                        #   only, because computing them requires 4-step Euler
                        #   integration to produce pred.
                        if is_fm and is_pattern_epoch:
                            pred = self._euler_sample_with_grad(inputs)

                        if (not is_fm) or is_pattern_epoch:
                            # pred exists here (UNet: always; FM: K-th epoch)
                            pearson_corr = self.unet.pearson_corr_torch(
                                pred, lst_cn)
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
                        else:
                            # FM non-pattern epoch: no pattern losses
                            l_pearson = torch.tensor(0.0, device=device)
                            l_cold = torch.tensor(0.0, device=device)
                            l_subgroup = torch.tensor(0.0, device=device)

                        # Spectral: K-th epoch for both architectures
                        if self.lambda_spectral > 0 and is_pattern_epoch:
                            l_spectral = self.lambda_spectral * compute_spectral_loss(
                                pred, lst_cn)
                        else:
                            l_spectral = torch.tensor(0.0, device=device)

                        l_total = (l_weighted_mse + l_weighted_flow
                                   + l_pearson + l_cold + l_subgroup
                                   + l_spectral)

                        l_total.backward()
                        torch.nn.utils.clip_grad_norm_(unet_params, max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.cn.parameters()) + list(self.loss_weights.parameters()),
                            max_norm=1.0)
                        unet_optimizer.step()
                        cn_optimizer.step()

                        acc['l_total'] += l_total.item()
                        acc['l_mse'] += l_mse.item()
                        acc['l_pearson'] += l_pearson.item()
                        acc['l_cn_own'] += l_cn_own.item()
                        acc['l_cold'] += l_cold.item()
                        acc['l_subgroup'] += (
                            l_subgroup.item() if torch.is_tensor(l_subgroup)
                            else l_subgroup)
                        acc['l_spectral'] += (
                            l_spectral.item() if torch.is_tensor(l_spectral)
                            else l_spectral)
                        acc['l_weighted_mse'] += l_weighted_mse.item()
                        acc['l_weighted_flow'] += l_weighted_flow.item()
                        acc['l_cn_shifted'] += l_cn_shifted.item()

                        # Metrics that require pred (not available for FM
                        # on non-pattern epochs)
                        if (not is_fm) or is_pattern_epoch:
                            acc['hard_cold_train'] += self._hard_cold_pct(
                                pred, era5_norm)
                            with torch.no_grad():
                                acc['l_mae'] += (pred - lst_cn).abs().mean().item()
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
                        if self.unet.architecture == 'flow_matching':
                            self.unet.flow_model.eval()
                        else:
                            self.unet.encoder.eval()
                            self.unet.decoder.eval()
                        test_mse_sum = 0.0
                        test_mae_sum = 0.0
                        n_test = 0
                        with torch.no_grad():
                            for batch in test_loader:
                                inputs_t, targets_t, _ = batch
                                inputs_t = inputs_t.to(device, non_blocking=True)
                                targets_t = targets_t.to(device, non_blocking=True)
                                pred_t = self._unet_forward(inputs_t)
                                test_mse_sum += self.unet.loss_fn(
                                    pred_t, targets_t).item()
                                test_mae_sum += (pred_t - targets_t).abs().mean().item()
                                n_test += 1
                        test_mse = test_mse_sum / max(n_test, 1)
                        test_mae = test_mae_sum / max(n_test, 1)
                        train_mae = acc['l_mae'] / max(n_batches, 1)
                        out_range = self._get_output_range_k()

                        print(f"\nepoch: {global_epoch}  [{phase_str}]")
                        print(f"  l_cn_own:   {avg('l_cn_own'):.6f}  "
                              f"mean_nll={nll_history[-1]:.4f}")
                        is_fm = (self.unet.architecture == 'flow_matching')
                        p_label = "l_cfm" if is_fm else "l_mse"
                        print(f"  {p_label}:      train={avg('l_mse'):.6f}  "
                              f"test_mse={test_mse:.6f}")
                        if not is_fm:
                            print(f"  metrics:    train_rmse={avg('l_mse')**0.5:.6f}  "
                                  f"test_rmse={test_mse**0.5:.6f}  "
                                  f"train_mae={train_mae:.6f}  "
                                  f"test_mae={test_mae:.6f}  "
                                  f"rmse/mae={test_mse**0.5/max(test_mae,1e-10):.3f}")
                            print(f"  physical:   train_rmse={avg('l_mse')**0.5*out_range:.2f}K  "
                                  f"test_rmse={test_mse**0.5*out_range:.2f}K  "
                                  f"train_mae={train_mae*out_range:.2f}K  "
                                  f"test_mae={test_mae*out_range:.2f}K")
                        else:
                            print(f"  metrics:    "
                                  f"test_rmse={test_mse**0.5:.6f}  "
                                  f"test_mae={test_mae:.6f}  "
                                  f"rmse/mae={test_mse**0.5/max(test_mae,1e-10):.3f}")
                            print(f"  physical:   "
                                  f"test_rmse={test_mse**0.5*out_range:.2f}K  "
                                  f"test_mae={test_mae*out_range:.2f}K")
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
                        if self.unet.architecture == 'flow_matching':
                            self.unet.flow_model.eval()
                        else:
                            self.unet.encoder.eval()
                            self.unet.decoder.eval()
                        self.cn.eval()

                        test_loss_sum = test_pearson_sum = hard_cold_test_sum = 0.0
                        test_mae_sum = 0.0
                        subgroup_info_test = {}
                        n_test = 0

                        with torch.no_grad():
                            for batch in test_loader:
                                inputs_t, targets_t, _ = batch
                                inputs_t = inputs_t.to(device, non_blocking=True)
                                targets_t = targets_t.to(device, non_blocking=True)
                                pred_t = self._unet_forward(inputs_t)
                                test_loss_sum += self.unet.loss_fn(
                                    pred_t, targets_t).item()
                                test_mae_sum += (pred_t - targets_t).abs().mean().item()
                                pearson_t = self.unet.pearson_corr_torch(
                                    pred_t, targets_t)
                                test_pearson_sum += (
                                    1 - torch.mean(pearson_t)).item()
                                hard_cold_test_sum += self._hard_cold_pct(
                                    pred_t, self._get_era5(inputs_t))
                                _, sg_info = compute_subgroup_robustness_loss(
                                    pred_t, targets_t, inputs_t)
                                if not subgroup_info_test:
                                    subgroup_info_test = sg_info
                                n_test += 1

                        test_loss = test_loss_sum / max(n_test, 1)
                        test_mae = test_mae_sum / max(n_test, 1)
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
                        primary_label = ("l_cfm" if self.unet.architecture
                                         == 'flow_matching' else "l_mse")
                        print(f"  losses:     {primary_label}={avg('l_mse'):.6f}  "
                              f"l_pearson={avg('l_pearson'):.6f}  "
                              f"l_cn_own={avg('l_cn_own'):.6f}  "
                              f"l_cold={avg('l_cold'):.6f}  "
                              f"l_subgroup={avg('l_subgroup'):.6f}  "
                              f"l_spectral={avg('l_spectral'):.6f}")
                        print(f"  weighted:   w_mse*L_mse={avg('l_weighted_mse'):.6f}  "
                              f"w_flow*L_shifted={avg('l_weighted_flow'):.6f}  "
                              f"w_mse={w_mse:.4f}  w_flow={w_flow:.6f}  "
                              f"s_mse={self.loss_weights.log_var_mse.item():.3f}  "
                              f"s_flow={self.loss_weights.log_var_flow.item():.3f}")
                        print(f"  shift:      l_cn_shifted={avg('l_cn_shifted'):.6f}  "
                              f"C={self.nll_shift_C:.4f}  "
                              f"floor={SHIFT_FLOOR}")
                        train_label = ("cfm" if self.unet.architecture
                                       == 'flow_matching' else "mse")
                        print(f"  {train_label}:        train={train_loss:.6f}  "
                              f"test_mse={test_loss:.6f}  "
                              f"ema_test={_ema_test:.6f}  "
                              f"ema_ratio={_ema_ratio:.3f}  "
                              f"test_pearson={test_pearson:.4f}")
                        train_mae = acc['l_mae'] / max(n_batches, 1)
                        out_range = self._get_output_range_k()
                        print(f"  metrics:    train_rmse={train_loss**0.5:.6f}  "
                              f"test_rmse={test_loss**0.5:.6f}  "
                              f"train_mae={train_mae:.6f}  "
                              f"test_mae={test_mae:.6f}  "
                              f"rmse/mae={test_loss**0.5/max(test_mae,1e-10):.3f}")
                        print(f"  physical:   train_rmse={train_loss**0.5*out_range:.2f}K  "
                              f"test_rmse={test_loss**0.5*out_range:.2f}K  "
                              f"train_mae={train_mae*out_range:.2f}K  "
                              f"test_mae={test_mae*out_range:.2f}K")
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
                        h.setdefault('l_spectral', []).append(
                            float(avg('l_spectral')))
                        h.setdefault('hard_cold_test_pct', []).append(
                            float(hard_cold_test))
                        h.setdefault('hard_cold_train_pct', []).append(
                            float(hard_cold_train))
                        h.setdefault('cn_pct_w_below_0.1', []).append(
                            float(cn_diag.get('pct_w_below_0.1', 0)))
                        h.setdefault('log_var_mse', []).append(
                            float(self.loss_weights.log_var_mse.item()))
                        h.setdefault('log_var_flow', []).append(
                            float(self.loss_weights.log_var_flow.item()))
                        h['_ema_test_mse'] = float(_ema_test)
                        h['_ema_ratio'] = float(_ema_ratio)
                        h['nr_epochs'] = global_epoch + 1

                        if test_loss < _best_test_mse:
                            _best_test_mse = test_loss
                            h['_best_test_mse'] = float(_best_test_mse)
                            ckpt = os.path.join(
                                model_folder, 'checkpoint_best_test_mse')
                            print(f"  * New best test_mse "
                                  f"{test_loss:.6f} -> {ckpt}")
                            self._save_checkpoint(
                                ckpt, unet_optimizer, cn_optimizer,
                                unet_scheduler, cn_scheduler, global_epoch)

                        if (global_epoch >= RATIO_WARMUP_EPOCHS and
                                _ema_ratio < _best_ema_ratio):
                            _best_ema_ratio = _ema_ratio
                            h['_best_train_test_ratio'] = float(_best_ema_ratio)
                            ckpt = os.path.join(
                                model_folder, 'checkpoint_best_ratio')
                            print(f"  * New best ema_ratio "
                                  f"{_ema_ratio:.3f} -> {ckpt}")
                            self._save_checkpoint(
                                ckpt, unet_optimizer, cn_optimizer,
                                unet_scheduler, cn_scheduler, global_epoch)

                        self.unet.history['nr_epochs'] = global_epoch + 1
                        self.unet.save(model_folder)
                        self._save_joint_state(
                            os.path.join(model_folder, 'joint_state'),
                            unet_optimizer, cn_optimizer,
                            unet_scheduler, cn_scheduler)

                # ---- Diagnostic images ----
                if (global_epoch % DIAG_IMAGE_INTERVAL == 0):
                    self._generate_diagnostic_images(
                        diag_dir, global_epoch, test_ds, device)

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

                epoch_sec = time.time() - epoch_start
                print(f"  time:       {epoch_sec:.1f}s  ({3600/epoch_sec:.1f} epochs/hr)")    
            
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

    def _generate_diagnostic_images(self, diag_dir, global_epoch, test_ds, device):
        # Randomly sample DIAG_N_BOXES from test set
        n_diag = min(DIAG_N_BOXES, len(test_ds))
        indices = torch.randperm(len(test_ds))[:n_diag]
        inputs_list, targets_list = [], []
        for idx in indices:
            inp, tgt, _ = test_ds[idx.item()]
            inputs_list.append(inp)
            targets_list.append(tgt)
        inputs_d = torch.stack(inputs_list).to(device)
        targets_d = torch.stack(targets_list).to(device)

        was_cn_corrections = self.cn.apply_corrections
        self.cn.apply_corrections = True
        if self.unet.architecture == 'flow_matching':
            self.unet.flow_model.eval()
        else:
            self.unet.encoder.eval()
            self.unet.decoder.eval()
        self.cn.eval()

        with torch.no_grad():
            lst_cn = self.cn(inputs_d, targets_d)
            pred = self._unet_forward(inputs_d)
            nll_map = self.cn.get_last_nll()
            weight_map = self.cn.get_last_weights()
            x_mean_map = self.cn.get_last_x_mean()

        _save_diagnostic_images(
            diag_dir, global_epoch, inputs_d, targets_d, lst_cn,
            pred, nll_map, weight_map, x_mean_map, self.cn,
            n_boxes=DIAG_N_BOXES)

        self.cn.apply_corrections = was_cn_corrections
        print(f"  [diag] Saved {n_diag} diagnostic images to {diag_dir}")

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
            'cn_type': 'flow_soft_blend',
            'cn_n_input_channels': self.cn.n_input_channels,
            'cn_base_channels': self._cn_base_channels,
            'cn_pretrain_min_epochs': self.cn_pretrain_min_epochs,
            'cn_pretrain_max_epochs': self.cn_pretrain_max_epochs,
            'cn_convergence_threshold': self.cn_convergence_threshold,
            'cn_convergence_window': self.cn_convergence_window,
            'lambda_cold': self.lambda_cold,
            'cold_threshold_k': self.cold_threshold_k,
            'lambda_subgroup': self.lambda_subgroup,
            'lambda_spectral': self.lambda_spectral,
            'spectral_every_k_epochs': self.spectral_every_k_epochs,
            'cn_lr': self.cn_lr,
            'epoch': epoch,
            'log_var_mse': self.loss_weights.log_var_mse.item(),
            'log_var_flow': self.loss_weights.log_var_flow.item(),
            'nll_shift_C': self.nll_shift_C,
            'flow_n_coupling_layers': self._flow_n_coupling_layers,
            'flow_coupling_hidden': self._flow_coupling_hidden,
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
        cn_base_channels=jc.get('cn_base_channels', 64),
        cn_pretrain_min_epochs=jc.get('cn_pretrain_min_epochs', 10),
        cn_pretrain_max_epochs=jc.get('cn_pretrain_max_epochs', 200),
        cn_convergence_threshold=jc.get('cn_convergence_threshold', 0.01),
        cn_convergence_window=jc.get('cn_convergence_window', 5),
        lambda_cold=jc.get('lambda_cold', 0.1),
        cold_threshold_k=jc.get('cold_threshold_k', 10.0),
        lambda_subgroup=jc.get('lambda_subgroup', 0.1),
        lambda_spectral=jc.get('lambda_spectral', 0.0),
        spectral_every_k_epochs=jc.get('spectral_every_k_epochs', 10),
        cn_lr=jc.get('cn_lr', 0.0001),
        flow_n_coupling_layers=jc.get('flow_n_coupling_layers', 4),
        flow_coupling_hidden=jc.get('flow_coupling_hidden', 64),
    )

    n_input_channels = jc.get('cn_n_input_channels', None)
    if n_input_channels is not None:
        joint._build_cn(n_input_channels)

    if 'log_var_mse' in jc:
        joint.loss_weights.log_var_mse.data = torch.tensor(jc['log_var_mse'])
    if 'log_var_flow' in jc:
        joint.loss_weights.log_var_flow.data = torch.tensor(jc['log_var_flow'])

    if 'nll_shift_C' in jc and jc['nll_shift_C'] is not None:
        joint.nll_shift_C = jc['nll_shift_C']

    cn_weights_path = os.path.join(model_folder, 'joint_state', 'cn.weights')
    if os.path.exists(cn_weights_path) and joint.cn is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        joint.cn.load_state_dict(
            torch.load(cn_weights_path, map_location=device))
        print(f"Loaded CN weights from {cn_weights_path}")
    elif joint.cn is None:
        print("Warning: CN not built yet (n_input_channels unknown), "
              "weights will load after train_joint() builds the CN")
    else:
        print("Warning: CN weights not found, starting CN from scratch")

    return joint
