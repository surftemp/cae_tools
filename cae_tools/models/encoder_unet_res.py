import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
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

class DownsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, num_residual_layers):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(output_channels) for _ in range(num_residual_layers)]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.residual_blocks(x)
        return x

class Encoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size, num_residual_layers=1):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()

        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            kernel_size = layer.get_kernel_size()
            stride = layer.get_stride()
            padding = layer.get_output_padding()

            self.encoder_blocks.append(DownsampleBlock(input_channels, output_channels, kernel_size, stride, padding, num_residual_layers))

        self.flatten = nn.Flatten(start_dim=1)

        (chan, y, x) = layers[-1].get_output_dimensions()

        self.encoder_lin = nn.Sequential(
            nn.Linear(chan * y * x, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, encoded_space_dim)
        )

    def forward(self, x):
        x_skip = []
        for block in self.encoder_blocks:
            x = block(x)
            x_skip.append(x)

        x = self.flatten(x)
        x = self.encoder_lin(x)
        x_skip.pop()  # remove the last layer's output, not used for skip connections
        return x, x_skip
