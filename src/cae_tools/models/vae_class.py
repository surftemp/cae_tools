import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


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
#         print(f"Residual block: output shape {out.shape}")
        
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, num_res_layers):
        super(Encoder, self).__init__()

        self.num_res_layers = num_res_layers
        self.encoder_blocks = nn.ModuleList()

        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            kernel_size = layer.get_kernel_size()
            stride = layer.get_stride()
            padding = layer.get_output_padding()

            # Downsampling block
            self.encoder_blocks.append(DownsampleBlock(input_channels, output_channels, kernel_size, stride, padding))

            # Residual blocks
            for _ in range(num_res_layers):
#                 print(f"number of residual layer is {num_res_layers}")
                self.encoder_blocks.append(ResidualBlock(output_channels))

        self.flatten = nn.Flatten(start_dim=1)

        (chan, y, x) = layers[-1].get_output_dimensions()
        self.fc_mu = nn.Linear(chan * y * x, encoded_space_dim)
        self.fc_logvar = nn.Linear(chan * y * x, encoded_space_dim)

    def forward(self, x):
        x_skip = []
        for i, block in enumerate(self.encoder_blocks):
#             print(f"shape of x before block {i} is {x.shape}")
            x = block(x)
#             print(f"shape of x after block {i} is {x.shape}")

            # Save skip connection after downsampling block
            if isinstance(block, DownsampleBlock):
                x_skip.append(x)
                
        x_skip.pop()  # remove the last layer's output, not used for skip connections
        x = self.flatten(x)
#         print(f"After flatten: shape {x.shape}")
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, x_skip

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std




class UpsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(UpsampleBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size,
                                                 stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# class Decoder(nn.Module):
#     def __init__(self, layers, encoded_space_dim, fc_size):
#         super(Decoder, self).__init__()

#         (chan, y, x) = layers[0].get_input_dimensions()
#         self.chan, self.y, self.x = chan, y, x

#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, fc_size),
#             nn.ReLU(True),
#             nn.Linear(fc_size, chan * y * x),
#         )

#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, y, x))

#         self.decoder_blocks = nn.ModuleList()
#         for layer in layers:
#             input_channels = layer.get_input_dimensions()[0]
#             output_channels = layer.get_output_dimensions()[0]
#             if layer != layers[-1]:
#                 self.decoder_blocks.append(UpsampleBlock(input_channels, output_channels,
#                                                          layer.get_kernel_size(), layer.get_stride(),
#                                                          layer.get_output_padding()))
#             else:
#                 self.decoder_blocks.append(                
#                                     nn.ConvTranspose2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
#                                    stride=layer.get_stride(), output_padding=layer.get_output_padding()))

#     def forward(self, x, x_skip):
#         x = self.decoder_lin(x)
#         x = self.unflatten(x)
#         x_skip = x_skip[::-1]  # Reverse to match decoder order

#         for i, block in enumerate(self.decoder_blocks):
#             x = block(x)
#             if i < len(x_skip):  # Concatenate skip connections
#                 x = torch.cat((x, x_skip[i]), 1)

#         x = torch.sigmoid(x)
#         return x

import torch
from torch import nn
import torch.nn.init as init



class Decoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size):
        super().__init__()

        (chan, y, x) = layers[0].get_input_dimensions()
        # Unpacking and setting as instance attributes
        (self.chan, self.y, self.x) = layers[0].get_input_dimensions()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, chan * y * x),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, y, x))

        decoder_layers = []
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            decoder_layers.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                   stride=layer.get_stride(), output_padding=layer.get_output_padding()))
            if layer != layers[-1]:
                decoder_layers.append(nn.BatchNorm2d(output_channels*2))
                decoder_layers.append(nn.ReLU(True))

        self.decoder_conv = nn.ModuleList(decoder_layers)

        self._initialize_weights()        


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # For the last Linear layer use Xavier if it is followed by a Sigmoid
                if module.out_features == self.chan * self.y * self.x: 
                    init.xavier_normal_(module.weight)
                else:
                    init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)        

    def forward(self, x, x_skip):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        #x = self.decoder_conv(x)
        x_skip = x_skip[::-1]  # reverse to match decoder order

        skip_idx = 0        
        for layer in self.decoder_conv:
            x = layer(x)
            # version 1, concatenate skip connections after transposed convolution:
            if isinstance(layer, nn.ConvTranspose2d) and skip_idx < len(x_skip):
                x = torch.cat((x, x_skip[skip_idx]), 1)
                skip_idx += 1            
        x = torch.sigmoid(x)
        return x
    
    

class VAE(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, encoded_space_dim, fc_size, num_res_layers):
        super(VAE, self).__init__()
        self.encoder = Encoder(encoder_layers, encoded_space_dim, num_res_layers)
        self.decoder = Decoder(decoder_layers, encoded_space_dim, fc_size)

    def forward(self, x):
        mu, logvar, x_skip = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        prediction = self.decoder(z, x_skip)
        return prediction, mu, logvar
