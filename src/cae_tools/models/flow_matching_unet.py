"""
Flow Matching UNet for conditional generation.

Implements Rectified Flow / Flow Matching (Lipman et al. 2022, Liu et al. 2022):
- Learns straight-line transport from noise to data
- 1-4 step inference via Euler integration
- Same UNet backbone as StandardEncoder/StandardDecoder but with timestep conditioning

Two variants:
  FlowMatchingUNet: flat input — all channels concatenated with noisy target
  ConditionedFlowMatchingUNet: spatial/scalar separation — spatial channels
    concatenated with noisy target, ERA5 scalars injected via ConditioningModule
    at configurable stages (same as ConditionedEncoder/Decoder)

Timestep is injected into each residual block via additive projection.
ERA5 conditioning is applied before each ResBlock at configured stages.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels):
    """Select number of groups for GroupNorm based on channel count."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0 and channels // g >= 1:
            return g
    return 1


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""

    def __init__(self, sinusoidal_dim=128, hidden_dim=256):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def sinusoidal_embedding(self, t):
        """Create sinusoidal positional embedding for timestep t ∈ [0, 1]."""
        half = self.sinusoidal_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, sinusoidal_dim)

    def forward(self, t):
        """
        Args:
            t: (B,) timestep values in [0, 1]
        Returns:
            (B, hidden_dim) timestep embedding
        """
        emb = self.sinusoidal_embedding(t)
        return self.mlp(emb)


class TimeConditionedResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning injection.

    Structure:
        GroupNorm → ReLU → Conv3x3 → (+time_proj) → GroupNorm → ReLU → Dropout → Conv3x3 + shortcut

    The timestep embedding is projected to the channel dimension and added
    after the first convolution (standard practice from DDPM/score-based models).
    """

    def __init__(self, in_channels, out_channels, time_dim, dropout_rate=0.0):
        super().__init__()
        num_groups_1 = _get_num_groups(in_channels)
        num_groups_2 = _get_num_groups(out_channels)

        # First conv path
        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Timestep projection: project time embedding to out_channels
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        # Second conv path
        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: (B, C, H, W) feature map
            t_emb: (B, time_dim) timestep embedding
        """
        h = self.norm1(x)
        h = F.relu(h)
        h = self.conv1(h)

        # Inject timestep: add projected embedding (broadcast over spatial dims)
        t = self.time_proj(t_emb)[:, :, None, None]  # (B, out_channels, 1, 1)
        h = h + t

        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class FlowMatchingUNet(nn.Module):
    """
    UNet with timestep conditioning for flow matching.

    Architecture follows the same design as StandardEncoder/StandardDecoder:
    - 4 encoder stages with MaxPool downsampling
    - Convolutional bridge
    - 4 decoder stages with bilinear upsampling
    - Skip connections with concatenation + ResBlock blending
    - 1×1 conv output

    All residual blocks receive timestep conditioning.

    Input: conditioning (cond_channels) concatenated with noisy target (target_channels)
    Output: predicted velocity field (target_channels)
    """

    def __init__(self, cond_channels=12, target_channels=1, base_channels=64,
                 time_embed_dim=256, dropout_rate=0.0):
        super().__init__()
        self.cond_channels = cond_channels
        self.target_channels = target_channels

        in_ch = cond_channels + target_channels  # e.g. 12 + 1 = 13
        ch = base_channels  # 64

        # Timestep embedding
        self.time_embed = TimestepEmbedder(sinusoidal_dim=128, hidden_dim=time_embed_dim)

        # ---- Encoder ----
        self.enc1 = TimeConditionedResidualBlock(in_ch, ch, time_embed_dim, dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = TimeConditionedResidualBlock(ch, ch * 2, time_embed_dim, dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = TimeConditionedResidualBlock(ch * 2, ch * 4, time_embed_dim, dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = TimeConditionedResidualBlock(ch * 4, ch * 8, time_embed_dim, dropout_rate)
        self.pool4 = nn.MaxPool2d(2, 2)

        # ---- Bridge ----
        self.bridge = TimeConditionedResidualBlock(ch * 8, ch * 8, time_embed_dim, dropout_rate)

        # ---- Decoder ----
        # Stage 4: upsample bridge → concat skip4 → ResBlock
        self.up_conv4 = nn.Conv2d(ch * 8, ch * 8, 1)
        self.dec4 = TimeConditionedResidualBlock(ch * 8 + ch * 8, ch * 4, time_embed_dim, dropout_rate)

        # Stage 3: upsample → concat skip3 → ResBlock
        self.up_conv3 = nn.Conv2d(ch * 4, ch * 4, 1)
        self.dec3 = TimeConditionedResidualBlock(ch * 4 + ch * 4, ch * 2, time_embed_dim, dropout_rate)

        # Stage 2: upsample → concat skip2 → ResBlock
        self.up_conv2 = nn.Conv2d(ch * 2, ch * 2, 1)
        self.dec2 = TimeConditionedResidualBlock(ch * 2 + ch * 2, ch, time_embed_dim, dropout_rate)

        # Stage 1: upsample → concat skip1 → ResBlock
        self.up_conv1 = nn.Conv2d(ch, ch, 1)
        self.dec1 = TimeConditionedResidualBlock(ch + ch, ch, time_embed_dim, dropout_rate)

        # Output: predict velocity field
        self.out_conv = nn.Conv2d(ch, target_channels, 1)

    def forward(self, x_noisy, conditioning, t):
        """
        Forward pass.

        Args:
            x_noisy: (B, target_channels, H, W) — noisy interpolant at time t
            conditioning: (B, cond_channels, H, W) — input features (ERA5, land cover, etc.)
            t: (B,) — timestep in [0, 1], where 0=noise, 1=data

        Returns:
            (B, target_channels, H, W) — predicted velocity field
        """
        # Compute timestep embedding
        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Concatenate conditioning with noisy target
        x = torch.cat([conditioning, x_noisy], dim=1)  # (B, cond_ch + target_ch, H, W)

        # ---- Encoder ----
        s1 = self.enc1(x, t_emb)                    # ch × H × W
        s2 = self.enc2(self.pool1(s1), t_emb)        # 2ch × H/2 × W/2
        s3 = self.enc3(self.pool2(s2), t_emb)        # 4ch × H/4 × W/4
        s4 = self.enc4(self.pool3(s3), t_emb)        # 8ch × H/8 × W/8

        # ---- Bridge ----
        b = self.bridge(self.pool4(s4), t_emb)       # 8ch × H/16 × W/16

        # ---- Decoder ----
        up4 = F.interpolate(b, size=s4.shape[2:], mode='bilinear', align_corners=False)
        up4 = self.up_conv4(up4)
        d4 = self.dec4(torch.cat([up4, s4], dim=1), t_emb)  # 4ch × H/8 × W/8

        up3 = F.interpolate(d4, size=s3.shape[2:], mode='bilinear', align_corners=False)
        up3 = self.up_conv3(up3)
        d3 = self.dec3(torch.cat([up3, s3], dim=1), t_emb)  # 2ch × H/4 × W/4

        up2 = F.interpolate(d3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        up2 = self.up_conv2(up2)
        d2 = self.dec2(torch.cat([up2, s2], dim=1), t_emb)  # ch × H/2 × W/2

        up1 = F.interpolate(d2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        up1 = self.up_conv1(up1)
        d1 = self.dec1(torch.cat([up1, s1], dim=1), t_emb)  # ch × H × W

        return self.out_conv(d1)  # target_channels × H × W


def flow_matching_loss(model, conditioning, target, device=None):
    """
    Compute the flow matching (conditional flow matching / rectified flow) loss.

    Linear interpolation path: x_t = (1 - t) * noise + t * target
    Target velocity:           v   = target - noise
    Loss:                      MSE(model(x_t, cond, t), v)

    Args:
        model: FlowMatchingUNet
        conditioning: (B, cond_channels, H, W) normalized input features
        target: (B, target_channels, H, W) normalized ground truth
        device: torch device

    Returns:
        loss: scalar MSE loss
    """
    B = conditioning.shape[0]

    # Sample random timesteps uniformly in [0, 1]
    t = torch.rand(B, device=device)

    # Sample noise from standard normal
    noise = torch.randn_like(target)

    # Linear interpolation: x_t = (1-t)*noise + t*data
    t_expand = t[:, None, None, None]  # (B, 1, 1, 1)
    x_t = (1.0 - t_expand) * noise + t_expand * target

    # Target velocity: data - noise (straight line from noise to data)
    velocity_target = target - noise

    # Predict velocity
    velocity_pred = model(x_t, conditioning, t)

    # MSE loss on velocity
    return F.mse_loss(velocity_pred, velocity_target)


@torch.no_grad()
def flow_matching_sample(model, conditioning, target_shape, num_steps=4, device=None):
    """
    Generate samples via Euler integration of the learned velocity field.

    Starting from pure noise at t=0, integrate to t=1 (clean data).

    Args:
        model: FlowMatchingUNet (in eval mode)
        conditioning: (B, cond_channels, H, W) normalized input features
        target_shape: tuple (B, target_channels, H, W)
        num_steps: number of Euler steps (default 4)
        device: torch device

    Returns:
        (B, target_channels, H, W) — predicted output (in normalized space)
    """
    B = target_shape[0]

    # Start from pure noise
    x = torch.randn(target_shape, device=device)

    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)

        # Predict velocity at current position and time
        v = model(x, conditioning, t)

        # Euler step
        x = x + dt * v

    return x


# =====================================================================
# Conditioned Flow Matching UNet
# =====================================================================

from .conditioned_unet import ConditioningModule, ALL_STAGES, _cond_extra


class ConditionedFlowMatchingUNet(nn.Module):
    """
    Flow Matching UNet with separate spatial/scalar conditioning.

    Three orthogonal conditioning axes:
      1. Spatial structure: concat(spatial, x_noisy) as encoder input
      2. Weather state: ERA5 scalars via ConditioningModule at configurable stages
      3. Flow time: sinusoidal timestep embedding at every ResBlock

    Architecture mirrors ConditionedEncoder/ConditionedDecoder but uses
    TimeConditionedResidualBlock for timestep injection.

    Args:
        spatial_channels: number of spatial input channels (e.g. 8)
        target_channels: number of target channels (1 for temperature)
        cond_dim: ERA5 conditioning vector dimension (e.g. 11)
        base_channels: base UNet channel count (default 64)
        time_embed_dim: timestep embedding dimension (default 256)
        dropout_rate: dropout for ResidualBlocks
        activation: activation function name (currently informational)
        inject_stages: set of stage names for ERA5 injection (default: all)
        cond_method: 'concat' or 'film' (default: 'concat')
    """

    def __init__(self, spatial_channels, target_channels, cond_dim,
                 base_channels=64, time_embed_dim=256, dropout_rate=0.0,
                 activation='silu', inject_stages=None, cond_method='concat'):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.target_channels = target_channels
        self.cond_dim = cond_dim
        self.cond_method = cond_method

        in_ch = spatial_channels + target_channels  # concat spatial + noisy target
        ch = base_channels

        if inject_stages is None:
            inject_stages = ALL_STAGES
        self.inject_stages = inject_stages

        # Timestep embedding (injected at every ResBlock)
        self.time_embed = TimestepEmbedder(sinusoidal_dim=128, hidden_dim=time_embed_dim)

        # Helper: create ConditioningModule for ERA5 at a given stage
        def make_cond(stage_name, feat_ch):
            if stage_name in self.inject_stages:
                return ConditioningModule(feat_ch, cond_dim, method=cond_method)
            return None

        def extra(stage_name):
            return _cond_extra(cond_dim, stage_name, self.inject_stages, cond_method)

        # ---- Encoder ----
        self.cond_e1 = make_cond('e1', in_ch)
        self.enc1 = TimeConditionedResidualBlock(
            in_ch + extra('e1'), ch, time_embed_dim, dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.cond_e2 = make_cond('e2', ch)
        self.enc2 = TimeConditionedResidualBlock(
            ch + extra('e2'), ch * 2, time_embed_dim, dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.cond_e3 = make_cond('e3', ch * 2)
        self.enc3 = TimeConditionedResidualBlock(
            ch * 2 + extra('e3'), ch * 4, time_embed_dim, dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.cond_e4 = make_cond('e4', ch * 4)
        self.enc4 = TimeConditionedResidualBlock(
            ch * 4 + extra('e4'), ch * 8, time_embed_dim, dropout_rate)
        self.pool4 = nn.MaxPool2d(2, 2)

        # ---- Bridge ----
        self.cond_bridge = make_cond('bridge', ch * 8)
        self.bridge = TimeConditionedResidualBlock(
            ch * 8 + extra('bridge'), ch * 8, time_embed_dim, dropout_rate)

        # ---- Decoder ----
        self.up_conv4 = nn.Conv2d(ch * 8, ch * 8, 1)
        self.cond_d4 = make_cond('d4', ch * 8 + ch * 8)
        self.dec4 = TimeConditionedResidualBlock(
            ch * 8 + ch * 8 + extra('d4'), ch * 4, time_embed_dim, dropout_rate)

        self.up_conv3 = nn.Conv2d(ch * 4, ch * 4, 1)
        self.cond_d3 = make_cond('d3', ch * 4 + ch * 4)
        self.dec3 = TimeConditionedResidualBlock(
            ch * 4 + ch * 4 + extra('d3'), ch * 2, time_embed_dim, dropout_rate)

        self.up_conv2 = nn.Conv2d(ch * 2, ch * 2, 1)
        self.cond_d2 = make_cond('d2', ch * 2 + ch * 2)
        self.dec2 = TimeConditionedResidualBlock(
            ch * 2 + ch * 2 + extra('d2'), ch, time_embed_dim, dropout_rate)

        self.up_conv1 = nn.Conv2d(ch, ch, 1)
        self.cond_d1 = make_cond('d1', ch + ch)
        self.dec1 = TimeConditionedResidualBlock(
            ch + ch + extra('d1'), ch, time_embed_dim, dropout_rate)

        # Output: predict velocity field
        self.out_conv = nn.Conv2d(ch, target_channels, 1)

    def _apply_cond(self, x, cond, cond_module):
        """Apply ERA5 conditioning module if it exists."""
        if cond_module is not None:
            return cond_module(x, cond)
        return x

    def forward(self, x_noisy, spatial, cond, t):
        """
        Args:
            x_noisy: (B, target_channels, H, W) — noisy interpolant at time t
            spatial: (B, spatial_channels, H, W) — spatial features
            cond: (B, cond_dim) — ERA5 conditioning scalars
            t: (B,) — timestep in [0, 1]

        Returns:
            (B, target_channels, H, W) — predicted velocity field
        """
        t_emb = self.time_embed(t)

        # Concatenate spatial input with noisy target
        x = torch.cat([spatial, x_noisy], dim=1)

        # ---- Encoder ----
        x = self._apply_cond(x, cond, self.cond_e1)
        s1 = self.enc1(x, t_emb)

        x = self._apply_cond(self.pool1(s1), cond, self.cond_e2)
        s2 = self.enc2(x, t_emb)

        x = self._apply_cond(self.pool2(s2), cond, self.cond_e3)
        s3 = self.enc3(x, t_emb)

        x = self._apply_cond(self.pool3(s3), cond, self.cond_e4)
        s4 = self.enc4(x, t_emb)

        # ---- Bridge ----
        x = self._apply_cond(self.pool4(s4), cond, self.cond_bridge)
        b = self.bridge(x, t_emb)

        # ---- Decoder ----
        up4 = F.interpolate(b, size=s4.shape[2:], mode='bilinear', align_corners=False)
        up4 = self.up_conv4(up4)
        x = self._apply_cond(torch.cat([up4, s4], dim=1), cond, self.cond_d4)
        d4 = self.dec4(x, t_emb)

        up3 = F.interpolate(d4, size=s3.shape[2:], mode='bilinear', align_corners=False)
        up3 = self.up_conv3(up3)
        x = self._apply_cond(torch.cat([up3, s3], dim=1), cond, self.cond_d3)
        d3 = self.dec3(x, t_emb)

        up2 = F.interpolate(d3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        up2 = self.up_conv2(up2)
        x = self._apply_cond(torch.cat([up2, s2], dim=1), cond, self.cond_d2)
        d2 = self.dec2(x, t_emb)

        up1 = F.interpolate(d2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        up1 = self.up_conv1(up1)
        x = self._apply_cond(torch.cat([up1, s1], dim=1), cond, self.cond_d1)
        d1 = self.dec1(x, t_emb)

        return self.out_conv(d1)


def conditioned_flow_matching_loss(model, spatial, cond, target, device=None):
    """
    Flow matching loss for ConditionedFlowMatchingUNet.

    Returns both the velocity MSE loss and a reconstructed target prediction
    (derived from the velocity prediction at no extra cost) for computing
    conventional auxiliary losses.

    Args:
        model: ConditionedFlowMatchingUNet
        spatial: (B, spatial_channels, H, W)
        cond: (B, cond_dim) ERA5 conditioning scalars
        target: (B, target_channels, H, W)
        device: torch device

    Returns:
        velocity_loss: scalar MSE loss on velocity
        pred: (B, target_channels, H, W) reconstructed target estimate
    """
    B = spatial.shape[0]
    t = torch.rand(B, device=device)
    noise = torch.randn_like(target)
    t_expand = t[:, None, None, None]
    x_t = (1.0 - t_expand) * noise + t_expand * target
    velocity_target = target - noise

    velocity_pred = model(x_t, spatial, cond, t)
    velocity_loss = F.mse_loss(velocity_pred, velocity_target)

    # Reconstruct target: x_1_hat = x_t + (1 - t) * v_pred
    # Free — no extra forward pass, gradients flow through velocity_pred
    pred = x_t + (1.0 - t_expand) * velocity_pred

    return velocity_loss, pred


@torch.no_grad()
def conditioned_flow_matching_sample(model, spatial, cond, target_shape,
                                      num_steps=4, device=None):
    """
    Euler integration for ConditionedFlowMatchingUNet.

    Args:
        model: ConditionedFlowMatchingUNet (in eval mode)
        spatial: (B, spatial_channels, H, W)
        cond: (B, cond_dim) ERA5 conditioning scalars
        target_shape: tuple (B, target_channels, H, W)
        num_steps: Euler integration steps (default 4)
        device: torch device

    Returns:
        (B, target_channels, H, W) — predicted output
    """
    B = target_shape[0]
    x = torch.randn(target_shape, device=device)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)
        v = model(x, spatial, cond, t)
        x = x + dt * v

    return x
