"""
Conditioned UNet: Standard UNet backbone with per-level conditioning injection.

Atmospheric/temporal scalars (ERA5 variables, sin/cos DOY) are concatenated
with spatial feature maps at every encoder and decoder stage. This allows
the 3x3 conv kernels to learn spatially-varying interactions between
atmospheric state and terrain features (e.g. wind × valley → cold pooling).

The conditioning vector is tiled to the spatial resolution at each level
and concatenated channel-wise before the ResidualBlock. Each ResidualBlock's
first conv therefore has (C_features + C_cond) input channels, learning
cross-terms between conditioning and spatial features directly.

Architecture details (v9 texture improvements):
- Stages 1-2 (full/half resolution) use TWO sequential ResidualBlocks each,
  increasing receptive field from 5x5 to 9x9 at the scales where parcel-level
  texture (2-5 pixel, 100-300m) needs to be generated.
- Stages 3-4 and bridge use single ResidualBlocks (low-frequency content only).
- Decoder uses 3x3 conv after bilinear upsampling (not 1x1), so spatial
  features can be created BEFORE skip concatenation rather than being locked
  into bilinear smoothness.

Interface matches StandardEncoder/StandardDecoder for drop-in use:
    encoder(spatial_input, cond) -> (encoded, skips)
    decoder(encoded, skips, cond) -> output

The conditioning vector can be raw scalar values (e.g. 11 floats) or
flattened 2x2 grids (e.g. 44 floats). No MLP is used — the conv layers
at each level learn the cross-variable interactions directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard_unet import ResidualBlock


class ConditionedEncoder(nn.Module):
    """
    UNet encoder with conditioning concatenation at every stage.

    At each stage, the conditioning vector is tiled to match the current
    spatial resolution and concatenated with the feature maps before the
    ResidualBlock.

    When n_res_blocks_hi=2, stages 1-2 use two sequential ResidualBlocks
    (9x9 receptive field) for parcel-scale texture capacity. When
    n_res_blocks_hi=1 (default), all stages use single blocks (backward
    compatible with existing checkpoints).

    Args:
        spatial_in_channels: number of spatial input channels
        cond_dim: dimension of the conditioning vector
        base_channels: base channel count (default 64)
        dropout_rate: dropout rate for ResidualBlocks
        activation: 'relu' or 'silu'
        n_res_blocks_hi: number of ResidualBlocks at stages 1-2 (1 or 2)
    """

    def __init__(self, spatial_in_channels, cond_dim, base_channels=64,
                 dropout_rate=0.0, activation='relu', n_res_blocks_hi=1):
        super().__init__()

        self.cond_dim = cond_dim
        self.n_res_blocks_hi = n_res_blocks_hi
        ch = base_channels

        if n_res_blocks_hi == 2:
            # Double blocks at stages 1-2
            self.stage1_block1 = ResidualBlock(
                spatial_in_channels + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)
            self.stage1_block2 = ResidualBlock(
                ch + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)

            self.stage2_block1 = ResidualBlock(
                ch + cond_dim, ch * 2,
                dropout_rate=dropout_rate, activation=activation)
            self.stage2_block2 = ResidualBlock(
                ch * 2 + cond_dim, ch * 2,
                dropout_rate=dropout_rate, activation=activation)
        else:
            # Single blocks (backward compatible)
            self.stage1 = ResidualBlock(
                spatial_in_channels + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)
            self.stage2 = ResidualBlock(
                ch + cond_dim, ch * 2,
                dropout_rate=dropout_rate, activation=activation)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stages 3-4 and bridge: always single block
        self.stage3 = ResidualBlock(
            ch * 2 + cond_dim, ch * 4,
            dropout_rate=dropout_rate, activation=activation)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = ResidualBlock(
            ch * 4 + cond_dim, ch * 8,
            dropout_rate=dropout_rate, activation=activation)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bridge = ResidualBlock(
            ch * 8 + cond_dim, ch * 8,
            dropout_rate=dropout_rate, activation=activation)

    def _tile_cond(self, cond, h, w):
        """Tile conditioning vector (B, cond_dim) to (B, cond_dim, H, W)."""
        return cond[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, x, cond):
        c = self._tile_cond(cond, x.shape[2], x.shape[3])

        if self.n_res_blocks_hi == 2:
            x = self.stage1_block1(torch.cat([x, c], dim=1))
            c = self._tile_cond(cond, x.shape[2], x.shape[3])
            s1 = self.stage1_block2(torch.cat([x, c], dim=1))
        else:
            s1 = self.stage1(torch.cat([x, c], dim=1))
        x = self.pool1(s1)

        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        if self.n_res_blocks_hi == 2:
            x = self.stage2_block1(torch.cat([x, c], dim=1))
            c = self._tile_cond(cond, x.shape[2], x.shape[3])
            s2 = self.stage2_block2(torch.cat([x, c], dim=1))
        else:
            s2 = self.stage2(torch.cat([x, c], dim=1))
        x = self.pool2(s2)

        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        s3 = self.stage3(torch.cat([x, c], dim=1))
        x = self.pool3(s3)

        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        s4 = self.stage4(torch.cat([x, c], dim=1))
        x = self.pool4(s4)

        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.bridge(torch.cat([x, c], dim=1))

        skips = [s1, s2, s3, s4]
        return x, skips


class ConditionedDecoder(nn.Module):
    """
    UNet decoder with conditioning concatenation at every stage.

    When n_res_blocks_hi=2, decoder stages 3-4 (50x50 and 100x100) use two
    sequential ResidualBlocks and 3x3 upconv for texture generation. When
    n_res_blocks_hi=1 (default), all stages use single blocks and 1x1 upconv
    (backward compatible with existing checkpoints).

    Args:
        out_channels: output channels (1 for temperature)
        cond_dim: conditioning vector dimension
        base_channels: must match encoder
        dropout_rate: dropout for ResidualBlocks
        output_activation: 'none', 'sigmoid', or 'tanh'
        activation: 'relu' or 'silu'
        n_res_blocks_hi: number of ResidualBlocks at high-res stages (1 or 2)
    """

    def __init__(self, out_channels=1, cond_dim=11, base_channels=64,
                 dropout_rate=0.0, output_activation='none',
                 activation='relu', n_res_blocks_hi=1):
        super().__init__()

        self.cond_dim = cond_dim
        self.output_activation = output_activation
        self.n_res_blocks_hi = n_res_blocks_hi
        ch = base_channels

        # Upconv kernel: 3x3 for double-block mode, 1x1 for single-block
        upk = 3 if n_res_blocks_hi == 2 else 1
        upp = 1 if n_res_blocks_hi == 2 else 0

        # Stage 1 (low-res, 12x12): always single block
        self.up_conv1 = nn.Conv2d(ch * 8, ch * 8, kernel_size=upk, padding=upp)
        self.dec_block1 = ResidualBlock(
            ch * 8 + ch * 8 + cond_dim, ch * 4,
            dropout_rate=dropout_rate, activation=activation)

        # Stage 2 (25x25): always single block
        self.up_conv2 = nn.Conv2d(ch * 4, ch * 4, kernel_size=upk, padding=upp)
        self.dec_block2 = ResidualBlock(
            ch * 4 + ch * 4 + cond_dim, ch * 2,
            dropout_rate=dropout_rate, activation=activation)

        # Stage 3 (50x50)
        self.up_conv3 = nn.Conv2d(ch * 2, ch * 2, kernel_size=upk, padding=upp)
        if n_res_blocks_hi == 2:
            self.dec_block3a = ResidualBlock(
                ch * 2 + ch * 2 + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)
            self.dec_block3b = ResidualBlock(
                ch + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)
        else:
            self.dec_block3 = ResidualBlock(
                ch * 2 + ch * 2 + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)

        # Stage 4 (100x100)
        self.up_conv4 = nn.Conv2d(ch, ch, kernel_size=upk, padding=upp)
        if n_res_blocks_hi == 2:
            self.dec_block4a = ResidualBlock(
                ch + ch + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)
            self.dec_block4b = ResidualBlock(
                ch + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)
        else:
            self.dec_block4 = ResidualBlock(
                ch + ch + cond_dim, ch,
                dropout_rate=dropout_rate, activation=activation)

        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def _upsample_and_concat(self, x, skip, up_conv):
        """Bilinear upsample x to match skip's spatial size, then concat."""
        x = up_conv(x)
        x = F.interpolate(
            x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def _tile_cond(self, cond, h, w):
        """Tile conditioning vector (B, cond_dim) to (B, cond_dim, H, W)."""
        return cond[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, x, skips, cond):
        s1, s2, s3, s4 = skips

        # Stage 1
        x = self._upsample_and_concat(x, s4, self.up_conv1)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.dec_block1(torch.cat([x, c], dim=1))

        # Stage 2
        x = self._upsample_and_concat(x, s3, self.up_conv2)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.dec_block2(torch.cat([x, c], dim=1))

        # Stage 3
        x = self._upsample_and_concat(x, s2, self.up_conv3)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        if self.n_res_blocks_hi == 2:
            x = self.dec_block3a(torch.cat([x, c], dim=1))
            c = self._tile_cond(cond, x.shape[2], x.shape[3])
            x = self.dec_block3b(torch.cat([x, c], dim=1))
        else:
            x = self.dec_block3(torch.cat([x, c], dim=1))

        # Stage 4
        x = self._upsample_and_concat(x, s1, self.up_conv4)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        if self.n_res_blocks_hi == 2:
            x = self.dec_block4a(torch.cat([x, c], dim=1))
            c = self._tile_cond(cond, x.shape[2], x.shape[3])
            x = self.dec_block4b(torch.cat([x, c], dim=1))
        else:
            x = self.dec_block4(torch.cat([x, c], dim=1))

        x = self.final_conv(x)

        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            x = torch.tanh(x)

        return x
