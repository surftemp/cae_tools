#    Copyright (C) 2023  National Centre for Earth Observation (NCEO)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from torch import nn
import torch.nn.init as init

class Encoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size, num_size_preserving_layers=1):
        super().__init__()

        self.num_size_preserving_layers = num_size_preserving_layers
        encoder_layers = []

        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            kernel_size = layer.get_kernel_size()
            stride = layer.get_stride()
            padding = layer.get_output_padding()
            
            # First Conv2D for downsampling
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                                            stride=stride,padding=padding))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))

            # Configurable number of size-preserving Conv2Ds
            for _ in range(num_size_preserving_layers):
                encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=3,
                                                stride=1,padding='same'))
                encoder_layers.append(nn.BatchNorm2d(output_channels))
                encoder_layers.append(nn.ReLU(True))

        self.encoder_cnn = nn.ModuleList(encoder_layers)
        self.flatten = nn.Flatten(start_dim=1)

        (chan, y, x) = layers[-1].get_output_dimensions()

        self.fc_mu = nn.Linear(chan * y * x, encoded_space_dim)
        self.fc_logvar = nn.Linear(chan * y * x, encoded_space_dim)

    def forward(self, x):
        x_skip = []
        layer_counter = 0

        for i, layer in enumerate(self.encoder_cnn):
#             print(f"Layer {i}: {layer.__class__.__name__}")
#             print(f"Input shape: {x.shape}")
            x = layer(x)
#             print(f"Output shape: {x.shape}")
            if isinstance(layer, nn.ReLU):
                if layer_counter == 0:
                    x_skip.append(x)
                else:
                    x = x + x_skip[-1]
                layer_counter += 1

            if layer_counter == self.num_size_preserving_layers + 1:  # 1 downsampling + n size-preserving layers
                layer_counter = 0
                
#         print(f"Shape before flatten: {x.shape}")  # Debugging line
        x = self.flatten(x)
#         print(f"Shape after flatten: {x.shape}")  # Debugging line        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        x_skip.pop()  # Remove the last layer's output, not used for skip connections        
        return mu, logvar, x_skip

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
