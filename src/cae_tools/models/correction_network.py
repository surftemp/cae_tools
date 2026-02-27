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
  - Residual correction enforces identity by default (zero-init final layer)
  - No hardcoded cold prior — CN learns where to correct from gradient signal
  - Local spatial gradient feature gives CN evidence of cloud edges
  - Uniform identity loss on ALL pixels — corrections must be justified by MSE
  - Pluggable: can be replaced with any architecture that satisfies the
    CorrectionNetworkBase interface (e.g. diffusion-based in future)

v2 changes (vs original):
  - Removed cold_threshold_norm and cold_prior_bias — CN discovers what to
    correct purely from training signal, not from hardcoded temperature rules
  - Identity loss now uniform across all pixels (was warm-only, giving a free
    pass to ~20% of pixels that happened to be colder than ERA5)
  - Added local gradient feature: max |pixel - neighbor| over 8-connected
    neighborhood, giving the CN spatial evidence of cloud edges (sharp 30-40K
    transitions) without telling it what threshold to use
  - Tighter max_mask_fraction (0.35 → 0.05) — only 0.01% of pixels in the
    training set are >10K colder than ERA5, so 5% is already very generous
  - Higher lambda_identity default (0.1 → 1.0) — CN needs strong resistance
    to gratuitous changes given how few pixels actually need correction
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

    No hardcoded cold prior — M starts at sigmoid(0) = 0.5 but Delta starts
    at 0 (zero-init), so the initial correction is zero everywhere. The CN
    must learn where to correct from the training gradient signal alone.

    Inputs to the CNN:
      - lst_raw_norm:     1 channel
      - era5_norm:        1 channel
      - local_max_grad:   1 channel (max |pixel - neighbor| over 8-connected)
      - static inputs:    elevation, land_cover, slope_magnitude, slope_direction
                          (indices specified via static_channel_indices)

    Total input channels = 3 + len(static_channel_indices)
    """

    def __init__(
        self,
        static_channel_indices,    # list of channel indices from main input tensor
        base_channels=32,          # width of CNN (keep small — CN should be lightweight)
        n_conv_layers=4,           # depth of CNN
        lambda_sparsity=0.01,      # weight for L1 sparsity on (M * Delta)
        lambda_identity=1.0,       # weight for identity preservation on ALL pixels
        max_mask_fraction=0.05,    # soft cap on mean(M) — penalty kicks in above this
        lambda_mask_cap=1.0,       # weight for mask cap penalty
        # Backward compatibility: accept but ignore cold_threshold_norm from old configs
        cold_threshold_norm=None,
    ):
        super().__init__()

        self.static_channel_indices = static_channel_indices
        self.lambda_sparsity = lambda_sparsity
        self.lambda_identity = lambda_identity
        self.max_mask_fraction = max_mask_fraction
        self.lambda_mask_cap = lambda_mask_cap

        # input: lst_raw + era5 + local_max_grad + static channels
        in_channels = 3 + len(static_channel_indices)

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

        # Store last computed losses and mask M for retrieval
        self._reg_loss = None
        self._identity_loss = None
        self._mask_cap_loss = None
        self._last_mask_m = None

    @staticmethod
    def _compute_local_max_grad(lst_raw_norm):
        """
        Compute max absolute difference to 8-connected neighbors for each pixel.

        Cloud edges produce sharp transitions (30-40K over 1 pixel) that are
        physically unlike any natural surface boundary. This gives the CN
        spatial evidence without a hardcoded threshold — it can learn what
        gradient magnitudes indicate cloud contamination.

        Args:
            lst_raw_norm: (B, 1, H, W) normalised LST

        Returns:
            (B, 1, H, W) max |pixel - neighbor| over 8 directions
        """
        # Reflect-pad to handle edges
        padded = F.pad(lst_raw_norm, (1, 1, 1, 1), mode='reflect')
        H, W = lst_raw_norm.shape[2], lst_raw_norm.shape[3]

        max_grad = torch.zeros_like(lst_raw_norm)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[:, :, 1+dy:1+dy+H, 1+dx:1+dx+W]
                max_grad = torch.max(max_grad, (lst_raw_norm - neighbor).abs())

        return max_grad

    def forward(self, inputs, lst_raw_norm, era5_norm):
        """
        Args:
            inputs:       (B, C, H, W) full input tensor (all channels)
            lst_raw_norm: (B, 1, H, W) normalised raw Landsat LST
            era5_norm:    (B, 1, H, W) normalised ERA5 (channel 3 of inputs)
        Returns:
            lst_corrected_norm: (B, 1, H, W) corrected normalised LST
        """
        # Extract static channels
        static = inputs[:, self.static_channel_indices, :, :]  # (B, n_static, H, W)

        # Compute local spatial gradient feature
        local_grad = self._compute_local_max_grad(lst_raw_norm)

        # Concatenate CN inputs: LST + ERA5 + gradient + static
        cn_input = torch.cat([lst_raw_norm, era5_norm, local_grad, static], dim=1)

        # Forward through CNN
        out = self.net(cn_input)  # (B, 2, H, W)
        delta = out[:, 0:1, :, :]          # (B, 1, H, W) — correction magnitude
        m_logit = out[:, 1:2, :, :]        # (B, 1, H, W) — mask logit

        # No cold prior bias — CN learns where to correct from gradient signal
        m = torch.sigmoid(m_logit)  # (B, 1, H, W) in [0,1]

        # Corrected LST
        lst_corrected_norm = lst_raw_norm + m * delta

        # Store M for MMD loss computation in training loop
        self._last_mask_m = m

        # Regularisation 1: L1 sparsity on the actual correction applied (M * Delta)
        # Encourages the CN to be conservative — only correct where necessary
        self._reg_loss = self.lambda_sparsity * (m * delta).abs().mean()

        # Regularisation 2: Identity preservation on ALL pixels (uniform)
        # Every correction has a cost. The CN must overcome this penalty through
        # improved l_main (MSE) to justify modifying any pixel. Cloud pixels
        # give the biggest MSE benefit (30K wrong → huge gradient), so they
        # naturally attract corrections. Clean pixels gain nothing from correction,
        # so the identity cost dominates and the CN leaves them alone.
        self._identity_loss = self.lambda_identity * (m * delta.abs()).mean()

        # Regularisation 3: Mask cap — soft quadratic penalty when mean(M) > max_mask_fraction
        # Prevents the CN from "correcting" the majority of pixels.
        mean_m = m.mean()
        self._mask_cap_loss = self.lambda_mask_cap * F.relu(mean_m - self.max_mask_fraction).pow(2)

        return lst_corrected_norm

    def get_regularisation_loss(self):
        if self._reg_loss is None:
            return torch.tensor(0.0)
        return self._reg_loss

    def get_identity_loss(self):
        if self._identity_loss is None:
            return torch.tensor(0.0)
        return self._identity_loss

    def get_mask_cap_loss(self):
        if self._mask_cap_loss is None:
            return torch.tensor(0.0)
        return self._mask_cap_loss

    def get_last_mask(self):
        return self._last_mask_m


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
# MMD local distributional consistency loss
# =============================================================================

def mmd_gaussian_rbf(x, y, bandwidth=None):
    """
    Compute Maximum Mean Discrepancy between 1D sample sets x and y
    using a Gaussian RBF kernel with median heuristic bandwidth.

    Args:
        x: (N,) tensor — corrected pixel values (where M > 0.5)
        y: (M,) tensor — untouched pixel values (where M < 0.5)
        bandwidth: if None, use median heuristic computed from combined samples

    Returns:
        scalar tensor — MMD^2 estimate (non-negative, 0 when distributions match)
    """
    x = x.unsqueeze(1)  # (N, 1)
    y = y.unsqueeze(1)  # (M, 1)

    if bandwidth is None:
        # Median heuristic: bandwidth = median of all pairwise distances
        combined = torch.cat([x, y], dim=0)  # (N+M, 1)
        dists = torch.cdist(combined, combined, p=2)  # (N+M, N+M)
        # Upper triangle only, excluding diagonal
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        median_dist = dists[mask].median()
        # Avoid division by zero or degenerate bandwidth
        bandwidth = median_dist.clamp(min=1e-6)

    def rbf_kernel(a, b, h):
        # (N, M) kernel matrix
        dists_sq = torch.cdist(a, b, p=2).pow(2)
        return torch.exp(-dists_sq / (2 * h * h))

    K_xx = rbf_kernel(x, x, bandwidth)
    K_yy = rbf_kernel(y, y, bandwidth)
    K_xy = rbf_kernel(x, y, bandwidth)

    N = x.shape[0]
    M = y.shape[0]

    # Unbiased MMD^2 estimator
    # Remove diagonal from K_xx and K_yy
    mmd_xx = (K_xx.sum() - K_xx.diag().sum()) / (N * (N - 1) + 1e-8)
    mmd_yy = (K_yy.sum() - K_yy.diag().sum()) / (M * (M - 1) + 1e-8)
    mmd_xy = K_xy.mean()

    return mmd_xx + mmd_yy - 2 * mmd_xy


def mmd_loss_batch(lst_cn, mask_m, lambda_mmd, min_samples=50):
    """
    Compute MMD loss over a batch of boxes.

    For each box, split pixels by mask M (learned soft mask):
      - corrected:  M > 0.5  (CN modified these)
      - untouched:  M < 0.5  (CN left these alone)

    MMD measures whether corrected pixel values are statistically consistent
    with untouched pixel values within the same box. If they are, the
    corrections are plausible members of the local LST field.

    Args:
        lst_cn:     (B, 1, H, W) CN-corrected LST (normalised)
        mask_m:     (B, 1, H, W) soft mask M in [0, 1]
        lambda_mmd: float weight
        min_samples: minimum pixels in each split to compute MMD (skip box if fewer)

    Returns:
        scalar tensor — mean MMD loss across valid boxes
    """
    B = lst_cn.shape[0]
    lst_flat  = lst_cn.view(B, -1)   # (B, H*W)
    mask_flat = mask_m.view(B, -1)   # (B, H*W)

    mmd_vals = []
    for b in range(B):
        corrected_mask = mask_flat[b] > 0.5
        untouched_mask = mask_flat[b] < 0.5

        n_corr    = corrected_mask.sum().item()
        n_untouch = untouched_mask.sum().item()

        # Skip box if either set too small for meaningful MMD
        if n_corr < min_samples or n_untouch < min_samples:
            continue

        corrected = lst_flat[b][corrected_mask]   # (n_corr,)
        untouched = lst_flat[b][untouched_mask]   # (n_untouch,)

        # Cap sample sizes for computational efficiency (large boxes are expensive)
        if n_corr > 500:
            idx = torch.randperm(n_corr, device=corrected.device)[:500]
            corrected = corrected[idx]
        if n_untouch > 500:
            idx = torch.randperm(n_untouch, device=untouched.device)[:500]
            untouched = untouched[idx]

        mmd_vals.append(mmd_gaussian_rbf(corrected, untouched))

    if len(mmd_vals) == 0:
        # No valid boxes this batch — return zero with gradient attached
        return lst_cn.sum() * 0.0

    return lambda_mmd * torch.stack(mmd_vals).mean()

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

    pred_k = pred_norm * (max_out - min_out) + min_out
    era5_k = era5_norm * (era5_max - era5_min) + era5_min

    delta_k = pred_k - era5_k

    cold_score = torch.sigmoid((-delta_k - cold_threshold_k) / temperature)

    return cold_score.mean()
