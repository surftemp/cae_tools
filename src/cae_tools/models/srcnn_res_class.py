import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding="same")
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding="same")
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

class SRCNN_RES(nn.Module):
    def __init__(self, num_channels, num_res_layers):
        super(SRCNN_RES, self).__init__()

        self.initial_conv = nn.Conv2d(num_channels, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(True)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_layers)]
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.relu(x)

        x = self.residual_blocks(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x