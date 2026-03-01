"""
Standard Residual UNet for regression downscaling.

Design follows modern climate downscaling UNet conventions:
- Residual blocks (two 3x3 convs + shortcut) at each encoder/decoder stage
- GroupNorm instead of BatchNorm (batch-independent, stable)
- Max pooling for downsampling (separates feature extraction from spatial reduction)
- Bilinear upsample + conv in decoder (no checkerboard artifacts from ConvTranspose)
- Post-skip conv blocks in decoder (blends encoder/decoder features before next stage)
- 1x1 conv as final output layer (clean channel reduction)
- Same padding throughout (padding=1 for 3x3 convs, preserves spatial dims within stage)

Handles non-power-of-2 input sizes (e.g., 100x100) by matching skip connection
spatial dimensions during upsampling.

Interface matches existing Encoder/Decoder for drop-in use with UNET training infrastructure:
    encoder(x) -> (encoded, skips)
    decoder(encoded, skips) -> output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Two 3x3 convolutions with GroupNorm and ReLU, plus a residual shortcut.
    
    GroupNorm → ReLU → Conv3x3 → GroupNorm → ReLU → Conv3x3
    + shortcut (1x1 conv if channels change, identity otherwise)
    """
    def __init__(self, in_channels, out_channels, num_groups=None, dropout_rate=0.0):
        super().__init__()
        
        # Auto-select num_groups: divisor of both in and out channels
        if num_groups is None:
            num_groups = self._select_num_groups(min(in_channels, out_channels))
        
        self.gn1 = nn.GroupNorm(self._select_num_groups(in_channels), in_channels)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.gn2 = nn.GroupNorm(self._select_num_groups(out_channels), out_channels)
        self.relu2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Shortcut: 1x1 conv if channel count changes, identity otherwise
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    @staticmethod
    def _select_num_groups(channels):
        """Select a reasonable number of groups for GroupNorm."""
        for g in [32, 16, 8, 4, 2, 1]:
            if channels % g == 0 and channels // g >= 1:
                return g
        return 1
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.gn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.gn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        return out + residual


class StandardEncoder(nn.Module):
    """
    Standard UNet encoder with residual blocks and max pooling.
    
    Each stage: ResidualBlock → (capture skip) → MaxPool2d
    
    Channel progression: in_channels → 64 → 128 → 256 → 512
    Spatial progression: H×W → H/2 → H/4 → H/8 → H/16
    
    Returns:
        encoded: tensor after bridge (512 channels, H/16 × W/16)
        skips: list of tensors [skip1, skip2, skip3, skip4] before pooling
    """
    def __init__(self, in_channels=12, base_channels=64, dropout_rate=0.0):
        super().__init__()
        
        ch = base_channels  # 64
        
        # Encoder stages
        self.stage1 = ResidualBlock(in_channels, ch, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.stage2 = ResidualBlock(ch, ch * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.stage3 = ResidualBlock(ch * 2, ch * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.stage4 = ResidualBlock(ch * 4, ch * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bridge (bottleneck) - convolutional, no FC
        self.bridge = ResidualBlock(ch * 8, ch * 8, dropout_rate=dropout_rate)
    
    def forward(self, x):
        # Stage 1: (B, 12, 100, 100) → (B, 64, 100, 100)
        s1 = self.stage1(x)
        x = self.pool1(s1)  # (B, 64, 50, 50)
        
        # Stage 2: (B, 64, 50, 50) → (B, 128, 50, 50)
        s2 = self.stage2(x)
        x = self.pool2(s2)  # (B, 128, 25, 25)
        
        # Stage 3: (B, 128, 25, 25) → (B, 256, 25, 25)
        s3 = self.stage3(x)
        x = self.pool3(s3)  # (B, 256, 12, 12)
        
        # Stage 4: (B, 256, 12, 12) → (B, 512, 12, 12)
        s4 = self.stage4(x)
        x = self.pool4(s4)  # (B, 512, 6, 6)
        
        # Bridge: (B, 512, 6, 6) → (B, 512, 6, 6)
        x = self.bridge(x)
        
        skips = [s1, s2, s3, s4]
        return x, skips


class StandardDecoder(nn.Module):
    """
    Standard UNet decoder with bilinear upsampling and post-skip residual blocks.
    
    Each stage: Bilinear upsample → Concatenate skip → ResidualBlock
    Final: 1×1 conv to output channels.
    
    No sigmoid/tanh — output is unconstrained. Apply activation externally if needed.
    
    Args:
        out_channels: number of output channels (1 for temperature)
        base_channels: must match encoder's base_channels
        output_activation: 'none', 'sigmoid', or 'tanh'
    """
    def __init__(self, out_channels=1, base_channels=64, dropout_rate=0.0,
                 output_activation='none'):
        super().__init__()
        self.output_activation = output_activation
        
        ch = base_channels  # 64
        
        # Decoder stages: upsample + concat skip + ResBlock
        # After concat, channels = decoder_channels + skip_channels
        
        # Up1: bridge(512) upsample, concat skip4(512) → 1024 → 256
        self.up_conv1 = nn.Conv2d(ch * 8, ch * 8, kernel_size=1)  # pre-upsample channel adjustment
        self.dec_block1 = ResidualBlock(ch * 8 + ch * 8, ch * 4, dropout_rate=dropout_rate)
        
        # Up2: 256 upsample, concat skip3(256) → 512 → 128
        self.up_conv2 = nn.Conv2d(ch * 4, ch * 4, kernel_size=1)
        self.dec_block2 = ResidualBlock(ch * 4 + ch * 4, ch * 2, dropout_rate=dropout_rate)
        
        # Up3: 128 upsample, concat skip2(128) → 256 → 64
        self.up_conv3 = nn.Conv2d(ch * 2, ch * 2, kernel_size=1)
        self.dec_block3 = ResidualBlock(ch * 2 + ch * 2, ch, dropout_rate=dropout_rate)
        
        # Up4: 64 upsample, concat skip1(64) → 128 → 64
        self.up_conv4 = nn.Conv2d(ch, ch, kernel_size=1)
        self.dec_block4 = ResidualBlock(ch + ch, ch, dropout_rate=dropout_rate)
        
        # Final 1×1 conv to output channels
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)
    
    def _upsample_and_concat(self, x, skip, up_conv):
        """Bilinear upsample x to match skip's spatial size, then concatenate."""
        x = up_conv(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)
    
    def forward(self, x, skips):
        """
        Args:
            x: bridge output (B, 512, H/16, W/16)
            skips: [s1, s2, s3, s4] from encoder (not reversed)
        """
        s1, s2, s3, s4 = skips
        
        # Stage 1: (B, 512, 6, 6) → upsample to 12×12, concat s4 → (B, 1024, 12, 12) → (B, 256, 12, 12)
        x = self._upsample_and_concat(x, s4, self.up_conv1)
        x = self.dec_block1(x)
        
        # Stage 2: (B, 256, 12, 12) → upsample to 25×25, concat s3 → (B, 512, 25, 25) → (B, 128, 25, 25)
        x = self._upsample_and_concat(x, s3, self.up_conv2)
        x = self.dec_block2(x)
        
        # Stage 3: (B, 128, 25, 25) → upsample to 50×50, concat s2 → (B, 256, 50, 50) → (B, 64, 50, 50)
        x = self._upsample_and_concat(x, s2, self.up_conv3)
        x = self.dec_block3(x)
        
        # Stage 4: (B, 64, 50, 50) → upsample to 100×100, concat s1 → (B, 128, 100, 100) → (B, 64, 100, 100)
        x = self._upsample_and_concat(x, s1, self.up_conv4)
        x = self.dec_block4(x)
        
        # Final: (B, 64, 100, 100) → (B, 1, 100, 100)
        x = self.final_conv(x)
        
        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            x = torch.tanh(x)
        # 'none': no activation, output unconstrained
        
        return x
