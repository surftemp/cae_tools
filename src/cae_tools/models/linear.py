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

from torch import nn


class Linear(nn.Module):
    """Original full linear model (flattens entire spatial grid).
    WARNING: For 100x100 inputs with 11 channels this creates a
    110,000 x 10,000 weight matrix (~1.1B parameters). Only usable
    for very small spatial grids.
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        (chan1, y1, x1) = input_shape
        (chan2, y2, x2) = output_shape
        self.linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(chan1 * y1 * x1, chan2 * y2 * x2),
            nn.Unflatten(dim=1, unflattened_size=(chan2, y2, x2))
        )

    def forward(self, x):
        return self.linear(x)


class PixelLinear(nn.Module):
    """Per-pixel linear regression via 1x1 convolution.

    At each pixel independently: takes the C input channel values and
    produces 1 output value via a learned linear combination + bias.

    Mathematically: y_{i,j} = sum_{c=1}^{C} w_c * x_{c,i,j} + b

    Parameters: C weights + 1 bias (e.g. 12 for 11 input channels).
    The same weights are applied at every spatial location.
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        (in_ch, _, _) = input_shape
        (out_ch, _, _) = output_shape
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PixelMLP(nn.Module):
    """Per-pixel nonlinear regression via stacked 1x1 convolutions.

    At each pixel independently: takes the C input channel values,
    passes them through hidden layers with ReLU, produces 1 output.

    Architecture: C -> 64 -> 32 -> 1  (all 1x1 convolutions)

    This captures nonlinear interactions between input channels
    (e.g. elevation x land_cover) but uses NO spatial context.
    Each pixel's prediction depends only on its own C channel values.

    Parameters: C*64 + 64 + 64*32 + 32 + 32*1 + 1
    For C=11: 704 + 64 + 2048 + 32 + 32 + 1 = 2,881 parameters.
    """

    def __init__(self, input_shape, output_shape, hidden_channels=(64, 32)):
        super().__init__()
        (in_ch, _, _) = input_shape
        (out_ch, _, _) = output_shape

        layers = []
        prev_ch = in_ch
        for h_ch in hidden_channels:
            layers.append(nn.Conv2d(prev_ch, h_ch, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            prev_ch = h_ch
        layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def build_linear_model(architecture, input_shape, output_shape):
    """Factory function to create the appropriate linear model variant.

    Args:
        architecture: one of 'full', 'pixel_linear', 'pixel_mlp'
        input_shape: (channels, height, width)
        output_shape: (channels, height, width)

    Returns:
        nn.Module
    """
    if architecture == 'full':
        return Linear(input_shape, output_shape)
    elif architecture == 'pixel_linear':
        return PixelLinear(input_shape, output_shape)
    elif architecture == 'pixel_mlp':
        return PixelMLP(input_shape, output_shape)
    else:
        raise ValueError(f"Unknown linear architecture: {architecture}. "
                         f"Choose from: full, pixel_linear, pixel_mlp")
