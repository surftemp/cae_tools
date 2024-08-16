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

class DownsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, num_res_layers):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(output_channels) for _ in range(num_res_layers)]
        )

    def forward(self, x,x_skip):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_skip.append(x)        
        x = self.pool(x)
        x = self.residual_blocks(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, output_padding):
        super(UpsampleBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size,
                                                 stride=stride, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(3*output_channels)
        self.relu = nn.ReLU(True)
        self.residual_block = ResidualBlock(3*output_channels)
        

    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat((x, skip), dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.residual_block(x)        
        return x

class Bridge(nn.Module):
    def __init__(self, input_channels):
        super(Bridge, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels * 2)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)  # Increase channels from input_channels to input_channels * 2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)  # Keep channels at input_channels * 2
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNET_RES(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, num_res_layers=1):
        super(UNET_RES, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Create encoder blocks
        for layer in encoder_layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            kernel_size = layer.get_kernel_size()
            stride = layer.get_stride()
            padding = layer.get_output_padding()
            self.encoder_blocks.append(DownsampleBlock(input_channels, output_channels, kernel_size, stride, padding, num_res_layers))

        # Create bridge
        bottleneck_channels = encoder_layers[-1].get_output_dimensions()[0]
        self.bridge = Bridge(bottleneck_channels)

        # Create decoder blocks
        for layer in decoder_layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            kernel_size = layer.get_kernel_size()
            stride = layer.get_stride()
            output_padding = layer.get_output_padding()
            self.decoder_blocks.append(UpsampleBlock(input_channels, output_channels,
                                                     kernel_size, stride, output_padding))
        input_to_last = decoder_layers[-1].get_output_dimensions()[0]
        input_to_last = 3*input_to_last
        self.final_conv = nn.Conv2d(input_to_last, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_skip = []
#         print(f"Input: {x.shape}")
        for i, block in enumerate(self.encoder_blocks):
            x = block(x,x_skip)
#             print(f"After encoder block {i},x: {x.shape}")
#             print(f"After encoder block {i},x_skip: {x_skip[i].shape}")

        x = self.bridge(x)
#         print(f"After bridge: {x.shape}")
#         x_skip.pop() # remove the last skip connection since it is the bottle neck input
        x_skip = x_skip[::-1]  # reverse to match decoder order

        for i, block in enumerate(self.decoder_blocks):
#             print(f"to be concate at {i}: x:{x.shape}, x_skip{x_skip[i].shape}")
            x = block(x, x_skip[i])
#             print(f"After decoder block {i}: {x.shape}")

        x = self.final_conv(x)
#         print(f"After final convolution: {x.shape}")
        x = self.sigmoid(x)
#         print(f"Output: {x.shape}")
        return x