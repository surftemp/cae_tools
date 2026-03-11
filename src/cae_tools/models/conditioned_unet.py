"""
Conditioned UNet: Standard UNet backbone with configurable conditioning injection.

Conditioning injection is controlled by two parameters:
- inject_stages: set of stage names where conditioning is applied
  Stage names: e1, e2, e3, e4, bridge (encoder), d4, d3, d2, d1 (decoder)
- cond_method: 'concat' or 'film'

Default (all stages, concat) reproduces the original v9 architecture exactly.
Bottleneck-only (inject_stages={'bridge'}) reduces ERA5 influence and tile
boundary artifacts.

Concat mode: tiles conditioning vector to spatial dims and concatenates
channel-wise before the ResidualBlock. The ResidualBlock's first conv
learns cross-terms between conditioning and spatial features.

FiLM mode (Feature-wise Linear Modulation): an MLP maps the conditioning
vector to per-channel scale and bias. Applied as x * scale + bias, so the
feature map dimensions are unchanged (no extra channels).

Interface matches StandardEncoder/StandardDecoder for drop-in use:
    encoder(spatial_input, cond) -> (encoded, skips)
    decoder(encoded, skips, cond) -> output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard_unet import ResidualBlock

ALL_STAGES = frozenset({'e1', 'e2', 'e3', 'e4', 'bridge', 'd4', 'd3', 'd2', 'd1'})


class LandCoverEmbedding(nn.Module):
    """
    Learnable embedding for categorical land cover channel.

    Replaces the single normalized land cover channel with a dense embedding
    vector per pixel. The input is de-normalized back to integer class indices
    before the embedding lookup.

    Args:
        num_classes: number of land cover classes (auto-detected from norm params)
        embed_dim: embedding dimension (replaces 1 channel with embed_dim channels)
        lc_min: minimum land cover value in physical space (from normalisation)
        lc_max: maximum land cover value in physical space (from normalisation)
        lc_channel_idx: index of land cover channel in spatial inputs (default: 0)
    """

    def __init__(self, num_classes, embed_dim, lc_min, lc_max, lc_channel_idx=0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.lc_channel_idx = lc_channel_idx
        self.register_buffer('lc_min', torch.tensor(float(lc_min)))
        self.register_buffer('lc_max', torch.tensor(float(lc_max)))
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, spatial):
        """
        Args:
            spatial: (B, C_spatial, H, W) with land cover at channel lc_channel_idx

        Returns:
            (B, C_spatial - 1 + embed_dim, H, W) with land cover channel replaced
            by embedding
        """
        idx = self.lc_channel_idx
        lc_norm = spatial[:, idx, :, :]  # (B, H, W)

        # De-normalize to physical integers
        lc_phys = lc_norm * (self.lc_max - self.lc_min) + self.lc_min
        lc_int = lc_phys.round().long().clamp(0, self.num_classes - 1)

        # Embed: (B, H, W) -> (B, H, W, embed_dim) -> (B, embed_dim, H, W)
        lc_embed = self.embedding(lc_int).permute(0, 3, 1, 2)

        # Remove original land cover channel and insert embedding
        other = torch.cat([spatial[:, :idx, :, :],
                           spatial[:, idx+1:, :, :]], dim=1)
        return torch.cat([lc_embed, other], dim=1)


class ConditioningModule(nn.Module):
    """
    Per-stage conditioning injection module.

    Args:
        feature_channels: number of feature map channels at this stage
        cond_dim: dimension of the conditioning vector
        method: 'concat' or 'film'
    """

    def __init__(self, feature_channels, cond_dim, method='concat'):
        super().__init__()
        self.feature_channels = feature_channels
        self.cond_dim = cond_dim
        self.method = method

        if method == 'concat':
            self._out_channels = feature_channels + cond_dim
        elif method == 'film':
            self._out_channels = feature_channels
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(cond_dim * 2, feature_channels * 2),
            )
            # Initialize scale near 1, bias near 0
            nn.init.ones_(self.mlp[-1].weight[:feature_channels].data.mul_(0.01))
            nn.init.zeros_(self.mlp[-1].bias[:feature_channels])
            nn.init.zeros_(self.mlp[-1].weight[feature_channels:])
            nn.init.zeros_(self.mlp[-1].bias[feature_channels:])
        else:
            raise ValueError(f"Unknown conditioning method: {method}")

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x, cond):
        """
        Args:
            x: (B, C, H, W) feature maps
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, out_channels, H, W) — out_channels = C + cond_dim for concat,
            C for film
        """
        if self.method == 'concat':
            c = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            return torch.cat([x, c], dim=1)
        else:  # film
            params = self.mlp(cond)  # (B, C*2)
            scale = params[:, :self.feature_channels, None, None]  # (B, C, 1, 1)
            bias = params[:, self.feature_channels:, None, None]
            return x * (1 + scale) + bias


def _cond_extra(cond_dim, stage_name, inject_stages, method):
    """Return extra input channels for a stage: cond_dim if concat+injected, else 0."""
    if stage_name in inject_stages and method == 'concat':
        return cond_dim
    return 0


class ConditionedEncoder(nn.Module):
    """
    UNet encoder with configurable per-stage conditioning injection.

    Args:
        spatial_in_channels: number of spatial input channels
        cond_dim: dimension of the conditioning vector
        base_channels: base channel count (default 64)
        dropout_rate: dropout rate for ResidualBlocks
        activation: 'relu' or 'silu'
        n_res_blocks_hi: number of ResidualBlocks at stages 1-2 (1 or 2)
        inject_stages: set of stage names to inject conditioning (default: all)
        cond_method: 'concat' or 'film' (default: 'concat')
    """

    def __init__(self, spatial_in_channels, cond_dim, base_channels=64,
                 dropout_rate=0.0, activation='relu', n_res_blocks_hi=1,
                 inject_stages=None, cond_method='concat',
                 lc_embed_dim=0, lc_num_classes=0, lc_min=0.0, lc_max=0.0,
                 lc_channel_idx=0):
        super().__init__()

        self.cond_dim = cond_dim
        self.n_res_blocks_hi = n_res_blocks_hi
        self.cond_method = cond_method
        self.lc_embed_dim = lc_embed_dim
        ch = base_channels

        if inject_stages is None:
            inject_stages = ALL_STAGES
        self.inject_stages = inject_stages

        # Land cover embedding: replaces 1 channel with embed_dim channels
        if lc_embed_dim > 0 and lc_num_classes > 0:
            self.lc_embedding = LandCoverEmbedding(
                num_classes=lc_num_classes, embed_dim=lc_embed_dim,
                lc_min=lc_min, lc_max=lc_max, lc_channel_idx=lc_channel_idx)
            effective_spatial_ch = spatial_in_channels - 1 + lc_embed_dim
        else:
            self.lc_embedding = None
            effective_spatial_ch = spatial_in_channels

        # Helper: create ConditioningModule for a stage if it's in inject_stages
        def make_cond(stage_name, feat_ch):
            if stage_name in self.inject_stages:
                return ConditioningModule(feat_ch, cond_dim, method=cond_method)
            return None

        # Extra channels added by concat conditioning
        def extra(stage_name):
            return _cond_extra(cond_dim, stage_name, self.inject_stages, cond_method)

        if n_res_blocks_hi == 2:
            # Double blocks at stages 1-2
            # e1 block1: input is spatial + maybe cond
            self.cond_e1_b1 = make_cond('e1', effective_spatial_ch)
            self.stage1_block1 = ResidualBlock(
                effective_spatial_ch + extra('e1'), ch,
                dropout_rate=dropout_rate, activation=activation)
            # e1 block2: input is ch + maybe cond
            self.cond_e1_b2 = make_cond('e1', ch)
            self.stage1_block2 = ResidualBlock(
                ch + extra('e1'), ch,
                dropout_rate=dropout_rate, activation=activation)

            # e2 block1
            self.cond_e2_b1 = make_cond('e2', ch)
            self.stage2_block1 = ResidualBlock(
                ch + extra('e2'), ch * 2,
                dropout_rate=dropout_rate, activation=activation)
            # e2 block2
            self.cond_e2_b2 = make_cond('e2', ch * 2)
            self.stage2_block2 = ResidualBlock(
                ch * 2 + extra('e2'), ch * 2,
                dropout_rate=dropout_rate, activation=activation)
        else:
            # Single blocks (backward compatible)
            self.cond_e1 = make_cond('e1', effective_spatial_ch)
            self.stage1 = ResidualBlock(
                effective_spatial_ch + extra('e1'), ch,
                dropout_rate=dropout_rate, activation=activation)

            self.cond_e2 = make_cond('e2', ch)
            self.stage2 = ResidualBlock(
                ch + extra('e2'), ch * 2,
                dropout_rate=dropout_rate, activation=activation)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stages 3-4 and bridge: always single block
        self.cond_e3 = make_cond('e3', ch * 2)
        self.stage3 = ResidualBlock(
            ch * 2 + extra('e3'), ch * 4,
            dropout_rate=dropout_rate, activation=activation)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.cond_e4 = make_cond('e4', ch * 4)
        self.stage4 = ResidualBlock(
            ch * 4 + extra('e4'), ch * 8,
            dropout_rate=dropout_rate, activation=activation)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.cond_bridge = make_cond('bridge', ch * 8)
        self.bridge = ResidualBlock(
            ch * 8 + extra('bridge'), ch * 8,
            dropout_rate=dropout_rate, activation=activation)

    def _apply_cond(self, x, cond, cond_module):
        """Apply conditioning module if it exists, otherwise return x unchanged."""
        if cond_module is not None:
            return cond_module(x, cond)
        return x

    def forward(self, x, cond):
        # Apply land cover embedding if enabled
        if self.lc_embedding is not None:
            x = self.lc_embedding(x)

        if self.n_res_blocks_hi == 2:
            x = self._apply_cond(x, cond, self.cond_e1_b1)
            x = self.stage1_block1(x)
            x = self._apply_cond(x, cond, self.cond_e1_b2)
            s1 = self.stage1_block2(x)
        else:
            x = self._apply_cond(x, cond, self.cond_e1)
            s1 = self.stage1(x)
        x = self.pool1(s1)

        if self.n_res_blocks_hi == 2:
            x = self._apply_cond(x, cond, self.cond_e2_b1)
            x = self.stage2_block1(x)
            x = self._apply_cond(x, cond, self.cond_e2_b2)
            s2 = self.stage2_block2(x)
        else:
            x = self._apply_cond(x, cond, self.cond_e2)
            s2 = self.stage2(x)
        x = self.pool2(s2)

        x = self._apply_cond(x, cond, self.cond_e3)
        s3 = self.stage3(x)
        x = self.pool3(s3)

        x = self._apply_cond(x, cond, self.cond_e4)
        s4 = self.stage4(x)
        x = self.pool4(s4)

        x = self._apply_cond(x, cond, self.cond_bridge)
        x = self.bridge(x)

        skips = [s1, s2, s3, s4]
        return x, skips


class ConditionedDecoder(nn.Module):
    """
    UNet decoder with configurable per-stage conditioning injection.

    Args:
        out_channels: output channels (1 for temperature)
        cond_dim: conditioning vector dimension
        base_channels: must match encoder
        dropout_rate: dropout for ResidualBlocks
        output_activation: 'none', 'sigmoid', or 'tanh'
        activation: 'relu' or 'silu'
        n_res_blocks_hi: number of ResidualBlocks at high-res stages (1 or 2)
        inject_stages: set of stage names to inject conditioning (default: all)
        cond_method: 'concat' or 'film' (default: 'concat')
    """

    def __init__(self, out_channels=1, cond_dim=11, base_channels=64,
                 dropout_rate=0.0, output_activation='none',
                 activation='relu', n_res_blocks_hi=1,
                 inject_stages=None, cond_method='concat'):
        super().__init__()

        self.cond_dim = cond_dim
        self.output_activation = output_activation
        self.n_res_blocks_hi = n_res_blocks_hi
        self.cond_method = cond_method
        ch = base_channels

        if inject_stages is None:
            inject_stages = ALL_STAGES
        self.inject_stages = inject_stages

        def make_cond(stage_name, feat_ch):
            if stage_name in self.inject_stages:
                return ConditioningModule(feat_ch, cond_dim, method=cond_method)
            return None

        def extra(stage_name):
            return _cond_extra(cond_dim, stage_name, self.inject_stages, cond_method)

        # Upconv kernel: 3x3 for double-block mode, 1x1 for single-block
        upk = 3 if n_res_blocks_hi == 2 else 1
        upp = 1 if n_res_blocks_hi == 2 else 0

        # d4 (low-res, 12x12): always single block
        # After upsample+concat: ch*8 (from bridge) + ch*8 (skip4) = ch*16
        self.up_conv1 = nn.Conv2d(ch * 8, ch * 8, kernel_size=upk, padding=upp)
        self.cond_d4 = make_cond('d4', ch * 8 + ch * 8)
        self.dec_block1 = ResidualBlock(
            ch * 8 + ch * 8 + extra('d4'), ch * 4,
            dropout_rate=dropout_rate, activation=activation)

        # d3 (25x25): always single block
        self.up_conv2 = nn.Conv2d(ch * 4, ch * 4, kernel_size=upk, padding=upp)
        self.cond_d3 = make_cond('d3', ch * 4 + ch * 4)
        self.dec_block2 = ResidualBlock(
            ch * 4 + ch * 4 + extra('d3'), ch * 2,
            dropout_rate=dropout_rate, activation=activation)

        # d2 (50x50)
        self.up_conv3 = nn.Conv2d(ch * 2, ch * 2, kernel_size=upk, padding=upp)
        if n_res_blocks_hi == 2:
            self.cond_d2_a = make_cond('d2', ch * 2 + ch * 2)
            self.dec_block3a = ResidualBlock(
                ch * 2 + ch * 2 + extra('d2'), ch,
                dropout_rate=dropout_rate, activation=activation)
            self.cond_d2_b = make_cond('d2', ch)
            self.dec_block3b = ResidualBlock(
                ch + extra('d2'), ch,
                dropout_rate=dropout_rate, activation=activation)
        else:
            self.cond_d2 = make_cond('d2', ch * 2 + ch * 2)
            self.dec_block3 = ResidualBlock(
                ch * 2 + ch * 2 + extra('d2'), ch,
                dropout_rate=dropout_rate, activation=activation)

        # d1 (100x100)
        self.up_conv4 = nn.Conv2d(ch, ch, kernel_size=upk, padding=upp)
        if n_res_blocks_hi == 2:
            self.cond_d1_a = make_cond('d1', ch + ch)
            self.dec_block4a = ResidualBlock(
                ch + ch + extra('d1'), ch,
                dropout_rate=dropout_rate, activation=activation)
            self.cond_d1_b = make_cond('d1', ch)
            self.dec_block4b = ResidualBlock(
                ch + extra('d1'), ch,
                dropout_rate=dropout_rate, activation=activation)
        else:
            self.cond_d1 = make_cond('d1', ch + ch)
            self.dec_block4 = ResidualBlock(
                ch + ch + extra('d1'), ch,
                dropout_rate=dropout_rate, activation=activation)

        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def _upsample_and_concat(self, x, skip, up_conv):
        """Bilinear upsample x to match skip's spatial size, then concat."""
        x = up_conv(x)
        x = F.interpolate(
            x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def _apply_cond(self, x, cond, cond_module):
        """Apply conditioning module if it exists, otherwise return x unchanged."""
        if cond_module is not None:
            return cond_module(x, cond)
        return x

    def forward(self, x, skips, cond):
        s1, s2, s3, s4 = skips

        # d4
        x = self._upsample_and_concat(x, s4, self.up_conv1)
        x = self._apply_cond(x, cond, self.cond_d4)
        x = self.dec_block1(x)

        # d3
        x = self._upsample_and_concat(x, s3, self.up_conv2)
        x = self._apply_cond(x, cond, self.cond_d3)
        x = self.dec_block2(x)

        # d2
        x = self._upsample_and_concat(x, s2, self.up_conv3)
        if self.n_res_blocks_hi == 2:
            x = self._apply_cond(x, cond, self.cond_d2_a)
            x = self.dec_block3a(x)
            x = self._apply_cond(x, cond, self.cond_d2_b)
            x = self.dec_block3b(x)
        else:
            x = self._apply_cond(x, cond, self.cond_d2)
            x = self.dec_block3(x)

        # d1
        x = self._upsample_and_concat(x, s1, self.up_conv4)
        if self.n_res_blocks_hi == 2:
            x = self._apply_cond(x, cond, self.cond_d1_a)
            x = self.dec_block4a(x)
            x = self._apply_cond(x, cond, self.cond_d1_b)
            x = self.dec_block4b(x)
        else:
            x = self._apply_cond(x, cond, self.cond_d1)
            x = self.dec_block4(x)

        x = self.final_conv(x)

        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            x = torch.tanh(x)

        return x
