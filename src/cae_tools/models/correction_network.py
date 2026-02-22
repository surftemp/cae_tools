"""
Correction Network (CN) for joint training with the main downscaling UNET.

The CN takes raw Landsat LST + ERA5 + static spatial channels as input and
produces a corrected Landsat LST that:
  - Is as close to the original as possible (identity constraint)
  - Reduces influence of cloud-contaminated pixels
  - Preserves spatial coherence

Architecture: Shallow residual CNN operating at full spatial resolution.
Output: lst_corrected = lst_raw + M * Delta
  where M (soft mask, 0-1) controls WHERE to correct
  and   Delta (correction field) controls HOW MUCH to correct

At inference time, the CN is discarded. Only the main UNET is used.
The main network input/output specification is completely unchanged.

Design:
  - Residual correction enforces identity by default
  - Hard threshold prior initialises M toward 1 for obvious cloud pixels
  - Pluggable: can be replaced with any architecture that satisfies the
    CorrectionNetworkBase interface (e.g. diffusion-based in future)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


# =============================================================================
# Base interface — swap CN architecture without changing joint training code
# =============================================================================

class CorrectionNetworkBase(ABC, nn.Module):
    """
    Abstract base for correction networks.

    Any replacement architecture (e.g. diffusion-based) must implement:
      forward(inputs, lst_raw_norm, era5_norm) -> lst_corrected_norm
      get_regularisation_loss() -> scalar tensor (L1 sparsity etc)

    inputs:       (B, C_static, H, W) static spatial channels
    lst_raw_norm: (B, 1, H, W) normalised raw Landsat LST
    era5_norm:    (B, 1, H, W) normalised ERA5 SKT (spatially broadcast)
    returns:      (B, 1, H, W) normalised corrected LST
    """

    @abstractmethod
    def forward(self, inputs, lst_raw_norm, era5_norm):
        pass

    @abstractmethod
    def get_regularisation_loss(self):
        """Return the most recently computed internal regularisation loss."""
        pass


# =============================================================================
# Shallow CNN correction network (default implementation)
# =============================================================================

class ConvCorrectionNetwork(CorrectionNetworkBase):
    """
    Lightweight CNN that produces:
      M     (B, 1, H, W) — soft mask in [0,1], where to apply correction
      Delta (B, 1, H, W) — correction field (how much to add)

    Corrected LST = lst_raw + M * Delta

    The hard threshold prior initialises M toward 1 for pixels where
    lst_raw - era5 < cold_threshold_norm (cloud-suspect pixels).
    This is a learned soft mask — the threshold is a prior, not a hard rule.

    Inputs to the CNN:
      - lst_raw_norm:  1 channel
      - era5_norm:     1 channel
      - static inputs: elevation, land_cover, slope_magnitude, slope_direction
                       (indices specified via static_channel_indices)

    Total input channels = 2 + len(static_channel_indices)
    """

    def __init__(
        self,
        static_channel_indices,    # list of channel indices from main input tensor
        base_channels=32,          # width of CNN (keep small — CN should be lightweight)
        n_conv_layers=4,           # depth of CNN
        cold_threshold_norm=0.3,   # normalised delta below which pixels are suspect
                                   # (0.3 in norm space ≈ -10K depending on norm range)
        lambda_sparsity=0.01,      # weight for L1 sparsity on (M * Delta)
    ):
        super().__init__()

        self.static_channel_indices = static_channel_indices
        self.cold_threshold_norm = cold_threshold_norm
        self.lambda_sparsity = lambda_sparsity

        # input: lst_raw + era5 + static channels
        in_channels = 2 + len(static_channel_indices)

        layers = []
        ch_in = in_channels
        for i in range(n_conv_layers - 1):
            layers += [
                nn.Conv2d(ch_in, base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, base_channels), base_channels),
                nn.ReLU(inplace=True),
            ]
            ch_in = base_channels

        # Final layer outputs 2 channels: Delta and pre-sigmoid M
        layers.append(nn.Conv2d(ch_in, 2, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

        # Initialise final layer close to zero so correction starts near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Store last computed regularisation loss for retrieval
        self._reg_loss = None

    def forward(self, inputs, lst_raw_norm, era5_norm):
        """
        Args:
            inputs:       (B, C, H, W) full input tensor (all 12 channels)
            lst_raw_norm: (B, 1, H, W) normalised raw Landsat LST
            era5_norm:    (B, 1, H, W) normalised ERA5 (channel 3 of inputs)
        Returns:
            lst_corrected_norm: (B, 1, H, W) corrected normalised LST
        """
        # Extract static channels
        static = inputs[:, self.static_channel_indices, :, :]  # (B, n_static, H, W)

        # Concatenate CN inputs
        cn_input = torch.cat([lst_raw_norm, era5_norm, static], dim=1)  # (B, C_cn, H, W)

        # Forward through CNN
        out = self.net(cn_input)  # (B, 2, H, W)
        delta = out[:, 0:1, :, :]          # (B, 1, H, W) — correction magnitude
        m_logit = out[:, 1:2, :, :]        # (B, 1, H, W) — mask logit

        # Hard threshold prior: bias mask toward 1 for cloud-suspect pixels
        # delta_norm = lst - era5 in normalised space
        delta_norm = lst_raw_norm - era5_norm
        # Pixels where delta_norm < -cold_threshold_norm are suspect
        # Add a positive bias to m_logit for those pixels
        cold_prior_bias = torch.where(
            delta_norm < -self.cold_threshold_norm,
            torch.ones_like(m_logit) * 2.0,   # sigmoid(2) ≈ 0.88 — lean toward correcting
            torch.zeros_like(m_logit)
        )
        m = torch.sigmoid(m_logit + cold_prior_bias)  # (B, 1, H, W) in [0,1]

        # Corrected LST
        lst_corrected_norm = lst_raw_norm + m * delta

        # Regularisation: L1 sparsity on the actual correction applied (M * Delta)
        # Encourages the CN to be conservative — only correct where necessary
        self._reg_loss = self.lambda_sparsity * (m * delta).abs().mean()

        return lst_corrected_norm

    def get_regularisation_loss(self):
        if self._reg_loss is None:
            return torch.tensor(0.0)
        return self._reg_loss


# =============================================================================
# Factory function — makes it easy to swap CN architecture later
# =============================================================================

def build_correction_network(cn_type='conv', **kwargs):
    """
    Factory for correction networks.
    cn_type: 'conv' (default) — ConvCorrectionNetwork
             'diffusion' — future implementation
    """
    if cn_type == 'conv':
        return ConvCorrectionNetwork(**kwargs)
    else:
        raise NotImplementedError(f"CN type '{cn_type}' not yet implemented. "
                                  f"Available: 'conv'")


# =============================================================================
# Differentiable soft cold pixel rate — cl2 loss term
# =============================================================================

def soft_cold_pixel_rate(
    pred_norm,          # (B, 1, H, W) normalised main network prediction
    era5_norm,          # (B, 1, H, W) normalised ERA5
    norm_params,        # normalisation_parameters dict from UNET
    cold_threshold_k=10.0,   # K — cold if pred < era5 - threshold
    temperature=0.5,         # sigmoid sharpness (lower = sharper, harder gradient)
):
    """
    Differentiable approximation of the fraction of predicted pixels that
    are implausibly cold (pred < ERA5 - cold_threshold_k).

    Uses sigmoid to approximate the step function, preserving gradients
    all the way back through pred_norm -> main network -> lst_cn -> CN weights.

    Returns scalar tensor (mean cold score across batch).
    """
    # Denormalise prediction and ERA5 to Kelvin
    min_out = norm_params['min_output']
    max_out = norm_params['max_output']
    pred_k = pred_norm * (max_out - min_out) + min_out

    era5_min = norm_params['inputs']['era5_skt']['min']
    era5_max = norm_params['inputs']['era5_skt']['max']
    era5_k = era5_norm * (era5_max - era5_min) + era5_min

    # delta = pred - era5 (negative = cold relative to ERA5)
    delta_k = pred_k - era5_k

    # soft cold score: 1 when delta << -threshold, 0 when delta >> -threshold
    # sigmoid((-delta_k - cold_threshold_k) / temperature)
    cold_score = torch.sigmoid((-delta_k - cold_threshold_k) / temperature)

    return cold_score.mean()
