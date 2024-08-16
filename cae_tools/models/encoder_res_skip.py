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

class Encoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size, num_size_preserving_layers=1):
        super().__init__()

        self.num_size_preserving_layers = num_size_preserving_layers
        encoder_layers = []
        
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            
            # First Conv2D for downsampling
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                            stride=layer.get_stride(),padding=layer.get_output_padding()))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))
            
            # Configurable number of size-preserving Conv2Ds
            for _ in range(num_size_preserving_layers):
                encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=1,
                                                stride=1))
                encoder_layers.append(nn.BatchNorm2d(output_channels))
                encoder_layers.append(nn.ReLU(True))

        self.encoder_cnn = nn.ModuleList(encoder_layers)
        self.flatten = nn.Flatten(start_dim=1)

        (chan, y, x) = layers[-1].get_output_dimensions()

        self.encoder_lin = nn.Sequential(
            nn.Linear(chan * y * x, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, encoded_space_dim)
        )

    def forward(self, x):
        x_skip = []
        layer_counter = 0
        
        for i, layer in enumerate(self.encoder_cnn):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                if layer_counter == 0:
                    x_skip.append(x)
                else:
                    x = x + x_skip[-1]
                layer_counter += 1
                
            if layer_counter == self.num_size_preserving_layers + 1:  # 1 downsampling + n size-preserving layers
                layer_counter = 0

        x = self.flatten(x)
        x = self.encoder_lin(x)
        x_skip.pop()  # Remove the last layer's output, not used for skip connections
        return x, x_skip
