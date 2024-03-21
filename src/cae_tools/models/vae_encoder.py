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

class VAE_Encoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size):
        super().__init__()
        
        # Convolutional layers
        encoder_layers = []
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                            stride=layer.get_stride()))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))
        self.encoder_cnn = nn.Sequential(*encoder_layers)

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        (chan, y, x) = layers[-1].get_output_dimensions()
        self.encoder_fc = nn.Linear(chan * y * x, fc_size)
        # self.fc_relu = nn.ReLU(True)
        self.fc_relu = nn.LeakyReLU(negative_slope=0.01)        

        # Output layers for mu and logvar
        self.fc_mu = nn.Linear(fc_size, encoded_space_dim)
        self.fc_logvar = nn.Linear(fc_size, encoded_space_dim)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_fc(x)
        x = self.fc_relu(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var
