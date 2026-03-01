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
    ResidualBlock. This means each ResidualBlock sees (C_features + cond_dim)
    input channels.

    Channel progression (with base_channels=64):
        Stage 1: (spatial_in + cond_dim) → 64
        Stage 2: (64 + cond_dim) → 128
        Stage 3: (128 + cond_dim) → 256
        Stage 4: (256 + cond_dim) → 512
        Bridge:  (512 + cond_dim) → 512

    Spatial progression: H×W → H/2 → H/4 → H/8 → H/16

    Args:
        spatial_in_channels: number of spatial input channels (e.g. 8 for
            land_cover, elevation, slope_mag, slope_dir, urban, suburban,
            albedo, hot_pattern)
        cond_dim: dimension of the conditioning vector (e.g. 11 raw scalars
            or 44 for flattened 2×2 grids)
        base_channels: base channel count for the UNet (default 64)
        dropout_rate: dropout rate for ResidualBlocks

    Returns:
        encoded: tensor after bridge (base_channels*8, H/16, W/16)
        skips: list [s1, s2, s3, s4] of skip connection tensors
    """

    def __init__(self, spatial_in_channels, cond_dim, base_channels=64,
                 dropout_rate=0.0):
        super().__init__()

        self.cond_dim = cond_dim
        ch = base_channels

        # Each stage's ResidualBlock receives features + tiled conditioning
        self.stage1 = ResidualBlock(
            spatial_in_channels + cond_dim, ch, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = ResidualBlock(
            ch + cond_dim, ch * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = ResidualBlock(
            ch * 2 + cond_dim, ch * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = ResidualBlock(
            ch * 4 + cond_dim, ch * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bridge = ResidualBlock(
            ch * 8 + cond_dim, ch * 8, dropout_rate=dropout_rate)

    def _tile_cond(self, cond, h, w):
        """Tile conditioning vector (B, cond_dim) to (B, cond_dim, H, W)."""
        return cond[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, x, cond):
        """
        Args:
            x: (B, spatial_in_channels, H, W) spatial input features
            cond: (B, cond_dim) conditioning vector
        """
        # Stage 1: (B, spatial_in + cond_dim, 100, 100) → (B, 64, 100, 100)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        s1 = self.stage1(torch.cat([x, c], dim=1))
        x = self.pool1(s1)  # (B, 64, 50, 50)

        # Stage 2: (B, 64 + cond_dim, 50, 50) → (B, 128, 50, 50)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        s2 = self.stage2(torch.cat([x, c], dim=1))
        x = self.pool2(s2)  # (B, 128, 25, 25)

        # Stage 3: (B, 128 + cond_dim, 25, 25) → (B, 256, 25, 25)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        s3 = self.stage3(torch.cat([x, c], dim=1))
        x = self.pool3(s3)  # (B, 256, 12, 12)

        # Stage 4: (B, 256 + cond_dim, 12, 12) → (B, 512, 12, 12)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        s4 = self.stage4(torch.cat([x, c], dim=1))
        x = self.pool4(s4)  # (B, 512, 6, 6)

        # Bridge: (B, 512 + cond_dim, 6, 6) → (B, 512, 6, 6)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.bridge(torch.cat([x, c], dim=1))

        skips = [s1, s2, s3, s4]
        return x, skips


class ConditionedDecoder(nn.Module):
    """
    UNet decoder with conditioning concatenation at every stage.

    At each stage: bilinear upsample → concatenate skip → concatenate
    tiled conditioning → ResidualBlock.

    The skip connections carry pure spatial features (no conditioning) from
    the encoder. Conditioning is re-injected fresh at each decoder level.

    Channel progression (base_channels=64):
        Stage 1: bridge(512) up + skip4(512) + cond → 1024+cond → 256
        Stage 2: 256 up + skip3(256) + cond → 512+cond → 128
        Stage 3: 128 up + skip2(128) + cond → 256+cond → 64
        Stage 4: 64 up + skip1(64) + cond → 128+cond → 64
        Final:   64 → out_channels (1×1 conv, no conditioning)

    Args:
        out_channels: output channels (1 for temperature)
        cond_dim: conditioning vector dimension
        base_channels: must match encoder
        dropout_rate: dropout for ResidualBlocks
        output_activation: 'none', 'sigmoid', or 'tanh'
    """

    def __init__(self, out_channels=1, cond_dim=11, base_channels=64,
                 dropout_rate=0.0, output_activation='none'):
        super().__init__()

        self.cond_dim = cond_dim
        self.output_activation = output_activation
        ch = base_channels

        # Decoder stages: upsample + concat skip + concat cond + ResBlock
        self.up_conv1 = nn.Conv2d(ch * 8, ch * 8, kernel_size=1)
        self.dec_block1 = ResidualBlock(
            ch * 8 + ch * 8 + cond_dim, ch * 4, dropout_rate=dropout_rate)

        self.up_conv2 = nn.Conv2d(ch * 4, ch * 4, kernel_size=1)
        self.dec_block2 = ResidualBlock(
            ch * 4 + ch * 4 + cond_dim, ch * 2, dropout_rate=dropout_rate)

        self.up_conv3 = nn.Conv2d(ch * 2, ch * 2, kernel_size=1)
        self.dec_block3 = ResidualBlock(
            ch * 2 + ch * 2 + cond_dim, ch, dropout_rate=dropout_rate)

        self.up_conv4 = nn.Conv2d(ch, ch, kernel_size=1)
        self.dec_block4 = ResidualBlock(
            ch + ch + cond_dim, ch, dropout_rate=dropout_rate)

        # Final 1×1 conv — no conditioning here, just channel reduction
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
        """
        Args:
            x: bridge output (B, 512, H/16, W/16)
            skips: [s1, s2, s3, s4] from encoder (not reversed)
            cond: (B, cond_dim) conditioning vector
        """
        s1, s2, s3, s4 = skips

        # Stage 1: upsample + skip4 + cond → (B, 1024+cond, 12, 12) → 256
        x = self._upsample_and_concat(x, s4, self.up_conv1)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.dec_block1(torch.cat([x, c], dim=1))

        # Stage 2: upsample + skip3 + cond → (B, 512+cond, 25, 25) → 128
        x = self._upsample_and_concat(x, s3, self.up_conv2)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.dec_block2(torch.cat([x, c], dim=1))

        # Stage 3: upsample + skip2 + cond → (B, 256+cond, 50, 50) → 64
        x = self._upsample_and_concat(x, s2, self.up_conv3)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.dec_block3(torch.cat([x, c], dim=1))

        # Stage 4: upsample + skip1 + cond → (B, 128+cond, 100, 100) → 64
        x = self._upsample_and_concat(x, s1, self.up_conv4)
        c = self._tile_cond(cond, x.shape[2], x.shape[3])
        x = self.dec_block4(torch.cat([x, c], dim=1))

        # Final: (B, 64, 100, 100) → (B, 1, 100, 100)
        x = self.final_conv(x)

        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            x = torch.tanh(x)

        return x
