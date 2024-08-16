import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class Encoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, num_size_preserving_layers=1):
        super(Encoder, self).__init__()

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
                                            stride=stride, padding=padding))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))

            # Configurable number of size-preserving Conv2Ds
            for _ in range(num_size_preserving_layers):
                encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=3,
                                                stride=1, padding='same'))
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
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        x_skip.pop()  # Remove the last layer's output, not used for skip connections        
        return mu, logvar, x_skip

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

#         self._initialize_weights()        


#     def _initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.ConvTranspose2d):
#                 init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#                 if module.bias is not None:
#                     init.constant_(module.bias, 0)
#             elif isinstance(module, nn.Linear):
#                 # For the last Linear layer use Xavier if it is followed by a Sigmoid
#                 if module.out_features == self.chan * self.y * self.x: 
#                     init.xavier_normal_(module.weight)
#                 else:
#                     init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#                 if module.bias is not None:
#                     init.constant_(module.bias, 0)
#             elif isinstance(module, nn.BatchNorm2d):
#                 init.constant_(module.weight, 1)
#                 init.constant_(module.bias, 0)        

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
    def __init__(self, encoder_layers, decoder_layers, encoded_space_dim, fc_size, num_size_preserving_layers=10):
        super(VAE, self).__init__()
        self.encoder = Encoder(encoder_layers, encoded_space_dim, num_size_preserving_layers)
        self.decoder = Decoder(decoder_layers, encoded_space_dim, fc_size)

    def forward(self, x):
        mu, logvar, x_skip = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        prediction = self.decoder(z, x_skip)
        return prediction, mu, logvar
