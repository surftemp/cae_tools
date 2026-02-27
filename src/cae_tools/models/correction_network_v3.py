"""
Correction Network v3 — Conditional Normalizing Flow

Learns p(lst | context) via conditional normalizing flow with exact likelihood.
Detection: NLL > threshold → likely contaminated.
Correction: replace with conditional mean (z=0 through inverse flow).
Hard gating: clean pixels pass through exactly, zero gradient leakage.

Standing instruction: no VGG/perceptual loss code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod


# =============================================================================
# Base interface
# =============================================================================

class CorrectionNetworkBase(ABC, nn.Module):

    @abstractmethod
    def forward(self, inputs, lst_raw_norm, era5_norm):
        pass

    @abstractmethod
    def get_cn_own_loss(self):
        pass

    @abstractmethod
    def get_diagnostics(self):
        pass


# =============================================================================
# U-Net backbone for full-box spatial context
# =============================================================================

class CNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        self.gn1 = nn.GroupNorm(self._select_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(self._select_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch
            else nn.Identity()
        )

    @staticmethod
    def _select_groups(channels):
        for g in [16, 8, 4, 2, 1]:
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.gn1(x))
        out = self.conv1(out)
        out = F.relu(self.gn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class CNBackbone(nn.Module):
    """3-stage U-Net. 100→50→25→12→25→50→100."""

    def __init__(self, in_channels, base_channels=32, dropout_rate=0.0):
        super().__init__()
        ch = base_channels
        self.enc1 = CNResBlock(in_channels, ch, dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = CNResBlock(ch, ch * 2, dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = CNResBlock(ch * 2, ch * 4, dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = CNResBlock(ch * 4, ch * 4, dropout_rate)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = CNResBlock(ch * 4 + ch * 4, ch * 2, dropout_rate)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = CNResBlock(ch * 2 + ch * 2, ch, dropout_rate)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = CNResBlock(ch + ch, ch, dropout_rate)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        b = self.bottleneck(self.pool3(s3))
        d3 = F.interpolate(self.up3(b), size=s3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = F.interpolate(self.up2(d3), size=s2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = F.interpolate(self.up1(d2), size=s1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return d1


# =============================================================================
# Affine coupling layer
# =============================================================================

class AffineCouplingLayer(nn.Module):
    """
    Scalar affine transform: z' = z * exp(s(context)) + t(context)
    s, t are per-pixel, predicted from backbone features.
    Invertible: z = (z' - t) * exp(-s)
    """

    def __init__(self, context_channels, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, context):
        st = self.net(context)
        s = st[:, 0:1, :, :].clamp(-3.0, 3.0)
        t = st[:, 1:2, :, :]
        z_out = z * torch.exp(s) + t
        return z_out, s  # s is log|det Jacobian|

    def inverse(self, z_out, context):
        st = self.net(context)
        s = st[:, 0:1, :, :].clamp(-3.0, 3.0)
        t = st[:, 1:2, :, :]
        return (z_out - t) * torch.exp(-s)


# =============================================================================
# Flow CN
# =============================================================================

class FlowCorrectionNetwork(CorrectionNetworkBase):
    """
    Conditional normalizing flow for p(lst | context).

    Hard gating: if NLL > threshold, replace with conditional mean.
    Otherwise exact passthrough. No soft blending.
    """

    def __init__(self, static_channel_indices, base_channels=32,
                 n_coupling_layers=4, coupling_hidden=32,
                 nll_correction_threshold=5.0, dropout_rate=0.0):
        super().__init__()
        self.static_channel_indices = static_channel_indices
        self.nll_correction_threshold = nll_correction_threshold

        # apply_corrections: set by joint trainer
        # False during pre-training, True during joint training
        self.apply_corrections = True

        in_channels = 2 + len(static_channel_indices)  # lst + era5 + statics
        self.backbone = CNBackbone(in_channels, base_channels, dropout_rate)
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(base_channels, coupling_hidden)
            for _ in range(n_coupling_layers)
        ])

        self._cn_own_loss = None
        self._diagnostics = {}

    def _flow_forward(self, lst, context):
        """Data → latent. Returns z and total log_det."""
        z = lst
        total_log_det = torch.zeros_like(lst)
        for layer in self.coupling_layers:
            z, log_det = layer(z, context)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def _flow_inverse(self, z, context):
        """Latent → data."""
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, context)
        return x

    def _log_prob(self, lst, context):
        """Exact log p(lst | context) via change of variables."""
        z, log_det = self._flow_forward(lst, context)
        log_p_base = -0.5 * (z.pow(2) + math.log(2 * math.pi))
        return log_p_base + log_det, z

    def forward(self, inputs, lst_raw_norm, era5_norm):
        static = inputs[:, self.static_channel_indices, :, :]
        cn_input = torch.cat([lst_raw_norm, era5_norm, static], dim=1)
        context = self.backbone(cn_input)

        # ---- CN own loss: negative log-likelihood ----
        log_p, z = self._log_prob(lst_raw_norm, context)
        # Raw NLL (unscaled). Scaling handled by learned log-variance in joint trainer.
        self._cn_own_loss = -log_p.mean()

        # ---- Correction: hard gating, only when enabled ----
        if self.apply_corrections:
            with torch.no_grad():
                nll = -log_p.detach()
                mask = (nll > self.nll_correction_threshold).float()

            # Conditional mean: z=0 through inverse flow
            conditional_mean = self._flow_inverse(
                torch.zeros_like(lst_raw_norm), context
            )

            # Hard gate: exact passthrough for clean, replace for contaminated
            # mask=0 → lst_raw_norm (untouched), mask=1 → conditional_mean
            lst_cn = (1 - mask) * lst_raw_norm + mask * conditional_mean
        else:
            lst_cn = lst_raw_norm
            mask = torch.zeros_like(lst_raw_norm)
            nll = -log_p.detach()

        # ---- Diagnostics ----
        with torch.no_grad():
            correction = (lst_cn - lst_raw_norm).detach()
            self._diagnostics = {
                'pct_active': mask.mean().item() * 100.0,
                'mean_correction': correction.abs().mean().item(),
                'max_correction': correction.abs().max().item(),
                'mean_delta': correction.mean().item(),
                'mean_nll': nll.mean().item(),
                'median_nll': nll.median().item(),
                'p99_nll': nll.quantile(0.99).item(),
                'max_nll': nll.max().item(),
            }

        return lst_cn

    def get_cn_own_loss(self):
        """Raw unscaled NLL. Joint trainer applies learned weighting."""
        return self._cn_own_loss if self._cn_own_loss is not None else torch.tensor(0.0)

    def get_regularisation_loss(self):
        return self.get_cn_own_loss()

    def get_diagnostics(self):
        return self._diagnostics
