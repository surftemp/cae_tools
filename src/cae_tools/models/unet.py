import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from cae_tools.models.standard_unet import StandardEncoder, StandardDecoder
from cae_tools.models.flow_matching_unet import FlowMatchingUNet, flow_matching_loss, flow_matching_sample
from torchvision import models
import torch.nn.functional as F


import numpy as np
import xarray as xr
import json
import os
import time
import signal

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from ..utils.model_database import ModelDatabase


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Encoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, fc_size, dropout_rate=0.1, use_fc=True,
                 latent_activation='relu'):
        super().__init__()
        self.use_fc = use_fc
        self.latent_activation = latent_activation

        encoder_layers = []
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                            stride=layer.get_stride(), padding=layer.get_output_padding()))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.Dropout(dropout_rate))  # Add dropout after ReLU

        self.encoder_cnn = nn.ModuleList(encoder_layers)

        (chan, y, x) = layers[-1].get_output_dimensions()
        if self.use_fc:
            self.flatten = nn.Flatten(start_dim=1)
            fc_layers = [
                nn.Linear(chan * y * x, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(fc_size, encoded_space_dim),
            ]
            # Configurable latent activation
            if latent_activation == 'relu':
                fc_layers.append(nn.ReLU(True))
            elif latent_activation == 'leaky_relu':
                fc_layers.append(nn.LeakyReLU(0.01, inplace=True))
            # 'none': no activation — latent values can be negative
            fc_layers.append(nn.Dropout(dropout_rate))
            self.encoder_lin = nn.Sequential(*fc_layers)
            self.bridge = None
        else:
            self.flatten = None
            self.encoder_lin = None
            # Conv bridge: two 3x3 convolutions preserving spatial dimensions
            self.bridge = nn.Sequential(
                nn.Conv2d(chan, chan, kernel_size=3, padding=1),
                nn.BatchNorm2d(chan),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(chan, chan, kernel_size=3, padding=1),
                nn.BatchNorm2d(chan),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        x_skip = []
        for layer in self.encoder_cnn:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x_skip.append(x)

        if self.use_fc:
            x = self.flatten(x)
            x = self.encoder_lin(x)
        else:
            x = self.bridge(x)

        x_skip.pop()  # remove the last layer's output, not used for skip connections
        return x, x_skip

class Decoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, fc_size, dropout_rate=0.1, use_fc=True, use_attention=True,
                 skip_mode='concat', skip_dropout=0.0, skip_scale=1.0, latent_activation='relu',
                 output_activation='sigmoid'):
        super().__init__()
        self.use_fc = use_fc
        self.use_attention = use_attention
        self.skip_mode = skip_mode
        self.skip_dropout = skip_dropout
        self.skip_scale = skip_scale
        self.output_activation = output_activation

        (chan, y, x) = layers[0].get_input_dimensions()
        self.chan, self.y, self.x = layers[0].get_input_dimensions()

        if self.use_fc:
            fc_layers = [
                nn.Linear(encoded_space_dim, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(fc_size, chan * y * x),
            ]
            if latent_activation == 'relu':
                fc_layers.append(nn.ReLU(True))
            elif latent_activation == 'leaky_relu':
                fc_layers.append(nn.LeakyReLU(0.01, inplace=True))
            fc_layers.append(nn.Dropout(dropout_rate))
            self.decoder_lin = nn.Sequential(*fc_layers)
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, y, x))
        else:
            self.decoder_lin = None
            self.unflatten = None

        decoder_layers = []
        self.attention_layers = nn.ModuleList()
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            decoder_layers.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                   stride=layer.get_stride(), padding=layer.get_output_padding()))
            if layer != layers[-1]:
                if self.use_attention:
                    self.attention_layers.append(ChannelAttention(output_channels))
                # BN size depends on skip mode
                if skip_mode == 'concat':
                    decoder_layers.append(nn.BatchNorm2d(output_channels * 2))
                else:  # 'add'
                    decoder_layers.append(nn.BatchNorm2d(output_channels))
                decoder_layers.append(nn.ReLU(True))
                decoder_layers.append(nn.Dropout(dropout_rate))  # Add dropout after ReLU

        self.decoder_conv = nn.ModuleList(decoder_layers)

    def forward(self, x, x_skip):
        if self.use_fc:
            x = self.decoder_lin(x)
            x = self.unflatten(x)
        x_skip = x_skip[::-1]  # reverse to match decoder order

        skip_idx = 0        
        for layer in self.decoder_conv:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d) and skip_idx < len(x_skip):
                if self.use_attention:
                    attention = self.attention_layers[skip_idx](x)
                    x = x * attention  # Apply attention

                # Get skip, apply scale and training dropout
                skip = x_skip[skip_idx]
                if self.skip_scale != 1.0:
                    skip = self.skip_scale * skip
                if self.training and self.skip_dropout > 0:
                    if torch.rand(1).item() < self.skip_dropout:
                        skip = torch.zeros_like(skip)

                # Join skip with decoder features
                if self.skip_mode == 'concat':
                    x = torch.cat((x, skip), 1)
                else:  # 'add'
                    x = x + skip

                skip_idx += 1            
        if self.output_activation == 'tanh':
            x = torch.tanh(x)
        elif self.output_activation == 'none':
            pass  # no activation, unconstrained output
        else:
            x = torch.sigmoid(x)
        return x 

    def forward(self, predicted, ground_truth):
        predicted_3channel = predicted.repeat(1, 3, 1, 1)
        ground_truth_3channel = ground_truth.repeat(1, 3, 1, 1)
        
        predicted_resized = self.resize_transform(predicted_3channel)
        ground_truth_resized = self.resize_transform(ground_truth_3channel)

        predicted_normalized = self.normalize_transform(predicted_resized)
        ground_truth_normalized = self.normalize_transform(ground_truth_resized)
        
        predicted_normalized = predicted_normalized.to(self.device)
        ground_truth_normalized = ground_truth_normalized.to(self.device)
        
        predicted_features = self.perceptual_encoder(predicted_normalized)
        ground_truth_features = self.perceptual_encoder(ground_truth_normalized)
        
        #  perceptual loss (MSE between VGG features)
        loss = nn.MSELoss()(predicted_features, ground_truth_features)

        return loss    


def augment_batch(inputs, targets, slope_dir_channel=7):
    """
    Apply full D4 symmetry augmentation: random 90° rotation + random H-flip.
    
    This covers all 8 symmetries of a square (identity, 3 rotations, 2 flips, 
    2 diagonal reflections), each with equal 1/8 probability.
    
    Only slope_direction (continuous azimuth, normalized from [-180°, 180°] to [0, 1])
    needs correction. All other channels are pure spatial fields where rearranging
    pixels preserves their values.
    
    Slope direction corrections (in normalized [0,1] space):
    - 90° CCW rotation (k times): norm → (norm - k/4) % 1.0
      (azimuth decreases by k*90° because grid north rotates)
    - H-flip: norm → 1.0 - norm
      (negates azimuth: east↔west)
    
    Args:
        inputs: (B, C, H, W) tensor on device
        targets: (B, 1, H, W) tensor on device  
        slope_dir_channel: index of slope_direction channel (default 7, None to skip)
    
    Returns:
        augmented (inputs, targets) tensors
    """
    B = inputs.shape[0]

    # Step 1: Random 90° rotation — k ∈ {0, 1, 2, 3} per sample
    k = torch.randint(0, 4, (B,), device=inputs.device)
    for ki in range(1, 4):
        mask = (k == ki)
        if mask.any():
            inputs[mask] = torch.rot90(inputs[mask], ki, [-2, -1])
            targets[mask] = torch.rot90(targets[mask], ki, [-2, -1])
            # Correct slope_direction: azimuth rotates by -ki*90°
            # In normalized space: norm → (norm - ki/4) % 1.0
            if slope_dir_channel is not None and slope_dir_channel < inputs.shape[1]:
                sd = inputs[mask, slope_dir_channel, :, :]
                inputs[mask, slope_dir_channel, :, :] = (sd - ki * 0.25) % 1.0

    # Step 2: Random H-flip (per-sample)
    h_mask = torch.rand(B, device=inputs.device) < 0.5
    if h_mask.any():
        inputs[h_mask] = inputs[h_mask].flip(-1)
        targets[h_mask] = targets[h_mask].flip(-1)
        # Correct slope_direction: azimuth → -azimuth → norm: 1.0 - norm
        if slope_dir_channel is not None and slope_dir_channel < inputs.shape[1]:
            inputs[h_mask, slope_dir_channel, :, :] = 1.0 - inputs[h_mask, slope_dir_channel, :, :]

    return inputs, targets


class UNET(BaseModel):
    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, dropout_rate=0.1, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None, lambda_l1=0.001, lambda_pearson=0,
                 checkpoint_interval=None, bottleneck_type='fc', use_attention=True,
                 skip_mode='concat', skip_dropout=0.0, skip_scale=1.0, latent_activation='relu',
                 output_activation='sigmoid', predict_delta=False, delta_reference_channel=None,
                 architecture='legacy', base_channels=64, flow_steps=4,
                 augment=False, slope_direction_channel=7):
        """
        Create a convolutional autoencoder general model

        :param normalise_input: whether the input variable should be normalised
        :param normalise_output: whether the output variable should be normalised
        :param batch_size: batch size for training
        :param nr_epochs: number of iterations for training
        :param test_interval: calculate test statistics every this many iterations
        :param encoded_dim_size: size of the latent encoding, in neurons
        :param fc_size: size of the fully connected layers that connect the latent layer to encoder and decoder stages
        :param lr: learning rate
        :param weight_decay: weight decay?
        :param use_gpu: use GPU if present
        :param conv_kernel_size: size of the convolutional kernel to use
        :param conv_stride: stride to use in convolutional layers
        :param conv_input_layer_count: number of input convolutional layers to use
        :param conv_output_layer_count: number of output convolutional layers to use
        :param database_path: path to optional tracking database
        :param checkpoint_interval: save checkpoint every N epochs (None to disable)
        :param bottleneck_type: 'fc' for FC bottleneck (default), 'conv' for fully convolutional UNET
        :param use_attention: whether to use channel attention on skip connections (default True)
        """
        super().__init__()
        self.normalise_input = normalise_input
        self.normalise_output = normalise_output
        self.normalisation_parameters = None
        self.input_shape = self.output_shape = None
        self.encoder = self.decoder = None
        self.discriminator = None
        self.batch_size = batch_size
        self.nr_epochs = nr_epochs
        self.test_interval = test_interval
        self.encoded_dim_size = encoded_dim_size
        self.fc_size = fc_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate 
        self.use_gpu = use_gpu
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_input_layer_count = conv_input_layer_count
        self.conv_output_layer_count = conv_output_layer_count
        self.spec = None
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs': 0}
        self.optim = None
        self.optim_D = None
        self.db = ModelDatabase(database_path) if database_path else None
        self.lambda_l1 = lambda_l1
        self.lambda_pearson = lambda_pearson
        self.checkpoint_interval = checkpoint_interval
        self.bottleneck_type = bottleneck_type
        self.use_attention = use_attention
        self.skip_mode = skip_mode
        self.skip_dropout = skip_dropout
        self.skip_scale = skip_scale
        self.latent_activation = latent_activation
        self.output_activation = output_activation
        self.predict_delta = predict_delta
        self.delta_reference_channel = delta_reference_channel
        self.architecture = architecture
        self.base_channels = base_channels
        self.flow_steps = flow_steps
        self.flow_model = None
        self.augment = augment
        self.slope_direction_channel = slope_direction_channel
        self.adversarial_loss = nn.BCELoss()
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return {
            "type": "UNET",
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "batch_size": self.batch_size,
            "test_interval": self.test_interval,
            "encoded_dim_size": self.encoded_dim_size,
            "fc_size": self.fc_size,
            "lr": self.lr,
            "lambda_pearson": self.lambda_pearson,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "normalise_input": self.normalise_input,
            "normalise_output": self.normalise_output,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_stride": self.conv_stride,
            "conv_input_layer_count": self.conv_input_layer_count,
            "conv_output_layer_count": self.conv_output_layer_count,
            "bottleneck_type": self.bottleneck_type,
            "use_attention": self.use_attention,
            "skip_mode": self.skip_mode,
            "skip_dropout": self.skip_dropout,
            "skip_scale": self.skip_scale,
            "latent_activation": self.latent_activation,
            "output_activation": self.output_activation,
            "predict_delta": self.predict_delta,
            "delta_reference_channel": self.delta_reference_channel,
            "architecture": self.architecture,
            "base_channels": self.base_channels,
            "flow_steps": self.flow_steps,
            "augment": self.augment,
            "slope_direction_channel": self.slope_direction_channel,
            "model_id": self.get_model_id()
        }

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype=torch.float32, requires_grad=True).to(real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size(), requires_grad=False).to(real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def __train_epoch(self, batches, device, n_critic=5):
        self.encoder.train()
        self.decoder.train()
#         self.discriminator.train()
        lambda_l1 = self.lambda_l1
        lambda_pearson = self.lambda_pearson
        flag_pearson = False
        train_loss = []
        train_pearson_loss=[]
        train_bias_loss = []
        train_d_loss = []

        for i, (low_res, high_res, labels) in enumerate(batches):
            # Move to GPU per-batch
            low_res = low_res.to(device)
            high_res = high_res.to(device)
#             valid = torch.ones((high_res.size(0), 1 ), requires_grad=False).to(low_res.device)
#             fake = torch.zeros((high_res.size(0), 1 ), requires_grad=False).to(low_res.device)

            self.optim.zero_grad()
            encoded_data, skip = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data, skip)
            
            mean_pred = torch.mean(decoded_data, dim=(2, 3))  # mean over spatial dimensions (x, y)
            mean_target = torch.mean(high_res, dim=(2, 3))    # mean over spatial dimensions (x, y)
            #bias_loss = torch.abs(mean_pred - mean_target).mean()

            mse_loss = self.loss_fn(decoded_data, high_res)
            pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
            pearson_loss = 1 - torch.mean(pearson_corr)
#             perceptual_loss = self.perceptual_loss_fn(decoded_data, high_res) 
#             pearson_loss = perceptual_loss            
            
            combined_loss = mse_loss + lambda_pearson * pearson_loss #+ 0.1*bias_loss
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)
            self.optim.step()
            train_loss.append(mse_loss.item())
            train_pearson_loss.append(pearson_loss.item())
            #train_bias_loss.append(bias_loss.item())

        mean_loss = np.mean(train_loss)
        mean_pearson_loss = np.mean(train_pearson_loss)

#         mean_bias_loss = np.mean(train_bias_loss)

        mean_bias_loss = 0

        mean_d_loss = 0
        return float(mean_loss), float(mean_pearson_loss), float(mean_bias_loss), float(mean_d_loss)

    def __train_epoch_from_loader(self, data_loader, device, n_critic=5):
        """Train epoch iterating DataLoader directly (no batch preloading)."""
        self.encoder.train()
        self.decoder.train()
        lambda_l1 = self.lambda_l1
        lambda_pearson = self.lambda_pearson
        train_loss = []
        train_pearson_loss = []

        for i, (low_res, high_res, labels) in enumerate(data_loader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            # Apply augmentation — DataLoader's collate_fn (torch.stack) already
            # allocates new tensors, so we don't need .clone() here.
            if self.augment:
                low_res, high_res = augment_batch(low_res, high_res,
                                                   slope_dir_channel=self.slope_direction_channel)

            self.optim.zero_grad()
            encoded_data, skip = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data, skip)

            mse_loss = self.loss_fn(decoded_data, high_res)
            pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
            pearson_loss = 1 - torch.mean(pearson_corr)

            combined_loss = mse_loss + lambda_pearson * pearson_loss
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)      
            self.optim.step()
            train_loss.append(mse_loss.item())
            train_pearson_loss.append(pearson_loss.item())

        mean_loss = np.mean(train_loss)
        mean_pearson_loss = np.mean(train_pearson_loss)
        mean_bias_loss = 0
        mean_d_loss = 0
        return float(mean_loss), float(mean_pearson_loss), float(mean_bias_loss), float(mean_d_loss)

    def __test_epoch_from_loader(self, data_loader, device, save_arr=None):
        """Test epoch iterating DataLoader directly (no batch preloading)."""
        test_loss = []
        test_pearson_loss = []
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ctr = 0
            for (low_res, high_res, labels) in data_loader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)
                pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
                pearson_loss = 1 - torch.mean(pearson_corr)
                test_pearson_loss.append(pearson_loss.detach().cpu().numpy())

                loss = self.loss_fn(decoded_data, high_res)
                test_loss.append(loss.detach().cpu().numpy())
                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

        mean_loss = np.mean(test_loss)
        mean_pearson_loss = np.mean(test_pearson_loss)
        mean_bias_loss = 0
        return float(mean_loss), float(mean_pearson_loss), float(mean_bias_loss)

    def __train_epoch_flow_matching(self, data_loader, device):
        """Train epoch for flow matching: predict velocity field."""
        self.flow_model.train()
        train_loss = []

        for i, (conditioning, target, labels) in enumerate(data_loader):
            conditioning = conditioning.to(device)  # (B, 12, H, W)
            target = target.to(device)              # (B, 1, H, W)

            # Apply augmentation (H-flip, V-flip with slope_direction correction)
            if self.augment:
                conditioning, target = augment_batch(conditioning, target,
                                                      slope_dir_channel=self.slope_direction_channel)

            self.optim.zero_grad()
            loss = flow_matching_loss(self.flow_model, conditioning, target, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.optim.step()
            train_loss.append(loss.item())

        return float(np.mean(train_loss)), 0.0, 0.0, 0.0

    def __test_epoch_flow_matching(self, data_loader, device, save_arr=None):
        """Test epoch for flow matching: run Euler integration and compute metrics on output."""
        self.flow_model.eval()
        test_loss = []
        test_pearson_loss = []
        test_velocity_loss = []

        with torch.no_grad():
            ctr = 0
            for (conditioning, target, labels) in data_loader:
                conditioning = conditioning.to(device)
                target = target.to(device)
                B = conditioning.shape[0]

                # Compute velocity MSE (same way as training for comparison)
                t = torch.rand(B, device=device)
                noise = torch.randn_like(target)
                t_expand = t[:, None, None, None]
                x_t = (1.0 - t_expand) * noise + t_expand * target
                velocity_target = target - noise
                velocity_pred = self.flow_model(x_t, conditioning, t)
                test_velocity_loss.append(F.mse_loss(velocity_pred, velocity_target).item())
                
                # Run inference: Euler integration from noise to prediction
                target_shape = (B, target.shape[1], target.shape[2], target.shape[3])
                predicted = flow_matching_sample(
                    self.flow_model, conditioning, target_shape,
                    num_steps=self.flow_steps, device=device
                )

                # MSE on final output (not velocity)
                loss = self.loss_fn(predicted, target)
                test_loss.append(loss.item())

                # Pearson correlation on final output
                pearson_corr = self.pearson_corr_torch(predicted, target)
                pearson_loss = 1 - torch.mean(pearson_corr)
                test_pearson_loss.append(pearson_loss.item())

                if save_arr is not None:
                    save_arr[ctr:ctr + B, :, :, :] = predicted.cpu()
                ctr += B

        mean_loss = np.mean(test_loss)
        mean_pearson_loss = np.mean(test_pearson_loss)
        mean_velocity_loss = np.mean(test_velocity_loss)
        return float(mean_loss), float(mean_pearson_loss), float(mean_velocity_loss)

    def __test_epoch(self, batches, device, save_arr=None):
        test_loss = []
        test_pearson_loss=[]
        test_bias_loss=[]
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for (low_res, high_res, labels) in batches:
                # Move to GPU per-batch
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)
                pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
                pearson_loss = 1 - torch.mean(pearson_corr)  
                test_pearson_loss.append(pearson_loss.detach().cpu().numpy())

#                 mean_pred = torch.mean(decoded_data, dim=(2, 3))  # mean over spatial dimensions (x, y)
#                 mean_target = torch.mean(high_res, dim=(2, 3))    # mean over spatial dimensions (x, y)
#                 bias_loss = torch.abs(mean_pred - mean_target).mean()  
#                 test_bias_loss.append(bias_loss.detach().cpu().numpy())

                loss = self.loss_fn(decoded_data, high_res)
                test_loss.append(loss.detach().cpu().numpy())
                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size
                
        mean_loss = np.mean(test_loss)
        mean_pearson_loss = np.mean(test_pearson_loss)
#         mean_bias_loss = np.mean(test_bias_loss)
        mean_bias_loss = 0
        return float(mean_loss), float(mean_pearson_loss),float(mean_bias_loss)

    def score(self, batches, save_arr):
        if self.architecture == 'flow_matching':
            self.flow_model.eval()
            device = next(self.flow_model.parameters()).device
            with torch.no_grad():
                ctr = 0
                for conditioning in batches:
                    B = conditioning.shape[0]
                    target_shape = (B, self.output_shape[0], self.output_shape[1], self.output_shape[2])
                    predicted = flow_matching_sample(
                        self.flow_model, conditioning, target_shape,
                        num_steps=self.flow_steps, device=device
                    )
                    save_arr[ctr:ctr + B, :, :, :] = predicted.cpu()
                    ctr += B
        else:
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                ctr = 0
                for input_data in batches:
                    encoded_data, skip = self.encoder(input_data)
                    decoded_data = self.decoder(encoded_data, skip)
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                    ctr += self.batch_size

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def train(self, input_variables, output_variable, training_ds, testing_ds, model_path="", training_paths="", testing_paths=""):
        train_ds = DSDataset(training_ds, input_variables, output_variable,
                             normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        self.set_input_spec(train_ds.get_input_spec())
        self.set_output_spec(train_ds.get_output_spec())

        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds = DSDataset(testing_ds, input_variables, output_variable,
                            normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        if not self.spec:
            self.spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                 output_size=(output_y, output_x), output_channels=output_chan,
                                 kernel_size=self.conv_kernel_size, stride=self.conv_stride,
                                 input_layer_count=self.conv_input_layer_count, output_layer_count=self.conv_output_layer_count)

        use_fc = (self.bottleneck_type == 'fc')
        if not self.encoder:
            if self.architecture == 'standard':
                self.encoder = StandardEncoder(in_channels=input_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate)
            else:
                self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate, use_fc=use_fc, latent_activation=self.latent_activation)
        if not self.decoder:
            if self.architecture == 'standard':
                self.decoder = StandardDecoder(out_channels=output_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, output_activation=self.output_activation)
            else:
                self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate, use_fc=use_fc, use_attention=self.use_attention, skip_mode=self.skip_mode, skip_dropout=self.skip_dropout, skip_scale=self.skip_scale, latent_activation=self.latent_activation, output_activation=self.output_activation)
#         if not self.discriminator:
#             self.discriminator = Discriminator(output_chan)  # Ensure discriminator input channels match output image channels
        
        fill_value = 0  
        fill = tuple([fill_value] * input_chan)
            
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),            # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),              # Randomly flip the image vertically
            transforms.RandomRotation(30, fill=fill),     # Randomly rotate the image within the range of -30 to +30 degrees, filling with the fill value
            transforms.RandomResizedCrop(100),            # Randomly crop the image and resize to 100x100
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds.transform = train_transform
        test_ds.transform = test_transform

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        print(f'Running on device: {device}')

        start = time.time()

        self.loss_fn = torch.nn.MSELoss()
        self.encoder.to(device)
        self.decoder.to(device)
#         self.discriminator.to(device)

        self.optim = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        epochs_already_done = self.history.get('nr_epochs', 0)
        if 'total_nr_epochs' not in self.history:
            self.history['total_nr_epochs'] = self.nr_epochs
        T_max = self.history['total_nr_epochs']
        epochs_this_job = self.nr_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=T_max, eta_min=1e-5)
        self._scheduler = scheduler

        # Restore optimizer/scheduler state if continuing
        if hasattr(self, '_saved_optimizer_state_path') and self._saved_optimizer_state_path:
            optimizer_state = torch.load(self._saved_optimizer_state_path, map_location=device)
            self.optim.load_state_dict(optimizer_state)
            self._saved_optimizer_state_path = None
        if hasattr(self, '_saved_scheduler_state_path') and self._saved_scheduler_state_path:
            scheduler_state = torch.load(self._saved_scheduler_state_path, map_location=device)
            scheduler.load_state_dict(scheduler_state)
            self._saved_scheduler_state_path = None

        self._sigterm_received = False
        def _handle_sigterm(signum, frame):
            print(f"\n[SIGTERM received] Finishing current epoch then saving...")
            self._sigterm_received = True
        signal.signal(signal.SIGTERM, _handle_sigterm)

        # Keep batches on CPU, move to GPU per-batch to avoid VRAM OOM
        train_batches = [(low_res, high_res, labels) for low_res, high_res, labels in train_loader]
        test_batches = [(low_res, high_res, labels) for low_res, high_res, labels in test_loader]

        try:
            for epoch in range(epochs_this_job):
                global_epoch = epochs_already_done + epoch
                train_loss, train_pearson_loss, train_bias_loss, train_d_loss = self.__train_epoch(train_batches, device)
                if global_epoch < T_max:
                    scheduler.step()
                if epoch % self.test_interval == 0:
                    test_loss, test_pearson_loss, test_bias_loss = self.__test_epoch(test_batches, device)
                    lr = self.get_lr(self.optim)
                    self.history["train_loss"].append(train_loss)
                    self.history["test_loss"].append(test_loss)
                    print(f"epoch: {global_epoch}, train_mse: {train_loss:.6f}, train_pearson_loss: {train_pearson_loss:.4f}, test_mse: {test_loss:.6f}, test_pearson_loss: {test_pearson_loss:.4f}")
                    print(f"learn rate: {lr:.6f}")

                # Save checkpoint every N epochs
                if self.checkpoint_interval and model_path and (global_epoch + 1) % self.checkpoint_interval == 0:
                    self.history['nr_epochs'] = global_epoch + 1
                    checkpoint_path = os.path.join(model_path, f"checkpoint_epoch_{global_epoch + 1}")
                    print(f"Saving checkpoint to {checkpoint_path}...")
                    self.save(checkpoint_path)

                if self._sigterm_received:
                    self.history['nr_epochs'] = global_epoch + 1
                    if model_path:
                        self.save(model_path)
                    break
                    
        except KeyboardInterrupt:
            print("Training interrupted. Performing cleanup...")
            # Save emergency checkpoint on interrupt
            if model_path:
                emergency_path = os.path.join(model_path, "checkpoint_interrupted")
                print(f"Saving emergency checkpoint to {emergency_path}...")
                self.history['nr_epochs'] += epoch + 1
                self.save(emergency_path)
        finally:
            end = time.time()
            elapsed = end - start

        if not self._sigterm_received:
            self.history['nr_epochs'] = epochs_already_done + epochs_this_job

        print("elapsed:" + str(elapsed))

        if self.db:
            self.db.add_training_result(self.get_model_id(), "UNET", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss, testing_paths, test_loss, self.get_parameters(), self.spec.save())
        if model_path:
            self.save(model_path)

        # pass over the training and test sets and calculate model metrics

        metrics = {}
        metrics["test"] = self.evaluate(test_ds, device)
        metrics["train"] = self.evaluate(train_ds, device)

        self.dump_metrics("Test Metrics", metrics["test"])
        self.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.get_model_id(), training_paths, testing_paths, metrics)

    def train_from_datasets(self, train_ds, test_ds, model_path="", training_paths="", testing_paths=""):
        """
        Train from pre-built PyTorch Dataset objects (e.g., PreprocessedDataset).
        
        This is faster than train() because it skips:
        - xarray loading/parsing
        - DSDataset creation and normalization computation
        
        Args:
            train_ds: PyTorch Dataset with get_input_spec(), get_output_spec(), get_normalisation_parameters()
            test_ds: PyTorch Dataset (will use train_ds normalisation parameters)
            model_path: folder to save the trained model
            training_paths: string of training file paths (for logging)
            testing_paths: string of testing file paths (for logging)
        """
        self.set_input_spec(train_ds.get_input_spec())
        self.set_output_spec(train_ds.get_output_spec())
        
        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)
        
        print(f"Training cases: {len(train_ds)}, Test cases: {len(test_ds)}")
        print(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")

        if not self.spec:
            self.spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                 output_size=(output_y, output_x), output_channels=output_chan,
                                 kernel_size=self.conv_kernel_size, stride=self.conv_stride,
                                 input_layer_count=self.conv_input_layer_count, output_layer_count=self.conv_output_layer_count)

        use_fc = (self.bottleneck_type == 'fc')
        if self.architecture == 'flow_matching':
            if not self.flow_model:
                self.flow_model = FlowMatchingUNet(
                    cond_channels=input_chan, target_channels=output_chan,
                    base_channels=self.base_channels, dropout_rate=self.dropout_rate
                )
                print(f"Flow Matching UNet: {sum(p.numel() for p in self.flow_model.parameters()):,} parameters")
                print(f"Inference steps: {self.flow_steps}")
        else:
            if not self.encoder:
                if self.architecture == 'standard':
                    self.encoder = StandardEncoder(in_channels=input_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate)
                else:
                    self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, dropout_rate=self.dropout_rate, use_fc=use_fc, latent_activation=self.latent_activation)
            if not self.decoder:
                if self.architecture == 'standard':
                    self.decoder = StandardDecoder(out_channels=output_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, output_activation=self.output_activation)
                else:
                    self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, dropout_rate=self.dropout_rate, use_fc=use_fc, use_attention=self.use_attention, skip_mode=self.skip_mode, skip_dropout=self.skip_dropout, skip_scale=self.skip_scale, latent_activation=self.latent_activation, output_activation=self.output_activation)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        print(f'Running on device: {device}')

        start = time.time()

        self.loss_fn = torch.nn.MSELoss()

        if self.architecture == 'flow_matching':
            self.flow_model.to(device)
            self.optim = torch.optim.AdamW(self.flow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.encoder.to(device)
            self.decoder.to(device)
            self.optim = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=self.weight_decay)

        # T_max must span the TOTAL training duration, not just this job's epochs.
        # If we set T_max = self.nr_epochs (per-job), the scheduler restarts its cosine
        # cycle on every job boundary, causing a massive LR discontinuity.
        # Solution: store total_nr_epochs in history and use it as T_max always.
        epochs_already_done = self.history.get('nr_epochs', 0)
        if 'total_nr_epochs' not in self.history:
            # Fresh training: total target = nr_epochs
            self.history['total_nr_epochs'] = self.nr_epochs
        # Always respect total_nr_epochs from history (set by fresh-start or CLI override)
        T_max = self.history['total_nr_epochs']
        epochs_this_job = self.nr_epochs  # how many epochs to run in this job

        print(f"Scheduler: T_max={T_max} (total), already done={epochs_already_done}, this job={epochs_this_job}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=T_max, eta_min=1e-5)
        self._scheduler = scheduler  # store reference so save() can access it

        # Restore optimizer state if available (for training continuation)
        if hasattr(self, '_saved_optimizer_state_path') and self._saved_optimizer_state_path:
            print(f"Restoring optimizer state from {self._saved_optimizer_state_path}...")
            optimizer_state = torch.load(self._saved_optimizer_state_path, map_location=device)
            self.optim.load_state_dict(optimizer_state)
            self._saved_optimizer_state_path = None

        # Restore scheduler state if available (for training continuation)
        if hasattr(self, '_saved_scheduler_state_path') and self._saved_scheduler_state_path:
            print(f"Restoring scheduler state from {self._saved_scheduler_state_path}...")
            scheduler_state = torch.load(self._saved_scheduler_state_path, map_location=device)
            scheduler.load_state_dict(scheduler_state)
            self._saved_scheduler_state_path = None

        # SIGTERM handler: SLURM sends SIGTERM ~30s before killing the job.
        # We catch it, set a flag, and save a clean checkpoint at the end of the current epoch.
        self._sigterm_received = False
        def _handle_sigterm(signum, frame):
            print(f"\n[SIGTERM received] Finishing current epoch then saving emergency checkpoint...")
            self._sigterm_received = True
        signal.signal(signal.SIGTERM, _handle_sigterm)

        # ---- Best-checkpoint tracking (persists across job boundaries via history) ----
        # We track two "optimal" points:
        #
        # 1. checkpoint_best_test_mse/
        #    Saved when EMA-smoothed test MSE hits a new minimum.
        #    Best absolute generalisation — use this for inference.
        #
        # 2. checkpoint_best_ratio/
        #    Saved when train/test ratio hits a new minimum.
        #    Diagnostic: where the model was least overfit relative to training loss.
        #
        # EMA smoothing (alpha=0.3) removes single-epoch noise spikes without
        # masking real trends. State is stored in history so it survives job boundaries.
        #
        # Both checkpoints are saved via self.save() so --continue-training works
        # identically from them as from any other checkpoint.
        EMA_ALPHA = 0.3
        _ema_test = self.history.get('_ema_test_mse', None)
        _best_ema_test = self.history.get('_best_ema_test_mse', float('inf'))
        _ema_ratio = self.history.get('_ema_ratio', None)
        _best_ema_ratio = self.history.get('_best_train_test_ratio', float('inf'))
        # Minimum epoch before ratio tracking starts. Before this point train loss
        # >> test loss (ratio < 1) which is meaningless, and the EMA needs a few
        # steps to warm up anyway.
        RATIO_WARMUP_EPOCHS = 50

        try:
            for epoch in range(epochs_this_job):
                global_epoch = epochs_already_done + epoch  # epoch number in the full training run
                if self.architecture == 'flow_matching':
                    train_loss, train_pearson_loss, train_bias_loss, train_d_loss = self.__train_epoch_flow_matching(train_loader, device)
                else:
                    train_loss, train_pearson_loss, train_bias_loss, train_d_loss = self.__train_epoch_from_loader(train_loader, device)
                if global_epoch < T_max:
                    scheduler.step()
                if epoch % self.test_interval == 0:
                    if self.architecture == 'flow_matching':
                        test_loss, test_pearson_loss, test_bias_loss = self.__test_epoch_flow_matching(test_loader, device)
                    else:
                        test_loss, test_pearson_loss, test_bias_loss = self.__test_epoch_from_loader(test_loader, device)
                    lr = self.get_lr(self.optim)
                    self.history["train_loss"].append(train_loss)
                    self.history["test_loss"].append(test_loss)

                    # Update EMA of test loss
                    if _ema_test is None:
                        _ema_test = test_loss
                    else:
                        _ema_test = EMA_ALPHA * test_loss + (1 - EMA_ALPHA) * _ema_test

                    ratio = test_loss / train_loss if train_loss > 0 else float('inf')

                    # EMA of ratio (same alpha as test MSE)
                    if _ema_ratio is None:
                        _ema_ratio = ratio
                    else:
                        _ema_ratio = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * _ema_ratio

                    if self.architecture == 'flow_matching':
                        print(f"epoch: {global_epoch}, train_velocity_mse: {train_loss:.6f}, test_velocity_mse: {test_bias_loss:.6f}, test_output_mse: {test_loss:.6f}, test_pearson_loss: {test_pearson_loss:.4f}, ema_test: {_ema_test:.6f}, ema_ratio: {_ema_ratio:.3f}")
                    else:
                        print(f"epoch: {global_epoch}, train_mse: {train_loss:.6f}, train_pearson_loss: {train_pearson_loss:.4f}, test_mse: {test_loss:.6f}, test_pearson_loss: {test_pearson_loss:.4f}, ema_test: {_ema_test:.6f}, ema_ratio: {_ema_ratio:.3f}")
                    print(f"learn rate: {lr:.6f}")

                    # Persist EMA state and bests in history for job-boundary survival
                    self.history['_ema_test_mse'] = _ema_test
                    self.history['_ema_ratio'] = _ema_ratio

                    # ---- Checkpoint: best absolute test performance ----
                    if model_path and _ema_test < _best_ema_test:
                        _best_ema_test = _ema_test
                        self.history['_best_ema_test_mse'] = _best_ema_test
                        self.history['_best_ema_test_epoch'] = global_epoch
                        self.history['nr_epochs'] = global_epoch + 1
                        best_test_path = os.path.join(model_path, "checkpoint_best_test_mse")
                        print(f"  ★ New best smoothed test MSE {_ema_test:.6f} at epoch {global_epoch} → saving {best_test_path}")
                        self.save(best_test_path)

                    # ---- Checkpoint: best train/test ratio (least overfit) ----
                    # Guards: epoch > warmup (EMA needs time to stabilise, early epochs
                    # have train >> test so ratio < 1 which is meaningless) AND
                    # ema_ratio > 1.0 (model must actually be overfitting for ratio to matter).
                    ratio_ready = global_epoch >= RATIO_WARMUP_EPOCHS and _ema_ratio > 1.0
                    if model_path and ratio_ready and _ema_ratio < _best_ema_ratio:
                        _best_ema_ratio = _ema_ratio
                        self.history['_best_train_test_ratio'] = _best_ema_ratio
                        self.history['_best_ratio_epoch'] = global_epoch
                        self.history['nr_epochs'] = global_epoch + 1
                        best_ratio_path = os.path.join(model_path, "checkpoint_best_ratio")
                        print(f"  ★ New best EMA ratio {_ema_ratio:.3f} at epoch {global_epoch} → saving {best_ratio_path}")
                        self.save(best_ratio_path)

                # Save checkpoint every N epochs
                if self.checkpoint_interval and model_path and (global_epoch + 1) % self.checkpoint_interval == 0:
                    self.history['nr_epochs'] = global_epoch + 1
                    checkpoint_path = os.path.join(model_path, f"checkpoint_epoch_{global_epoch + 1}")
                    print(f"Saving checkpoint to {checkpoint_path}...")
                    self.save(checkpoint_path)

                # SIGTERM received: save and exit cleanly so SLURM can resubmit
                if self._sigterm_received:
                    print(f"[SIGTERM] Saving emergency checkpoint at global epoch {global_epoch + 1}...")
                    self.history['nr_epochs'] = global_epoch + 1
                    if model_path:
                        emergency_path = os.path.join(model_path, f"checkpoint_epoch_{global_epoch + 1}_sigterm")
                        self.save(emergency_path)
                        # Also overwrite the main model folder so --model-folder continues from here
                        self.save(model_path)
                    break

        except KeyboardInterrupt:
            print("Training interrupted. Performing cleanup...")
            if model_path:
                emergency_path = os.path.join(model_path, "checkpoint_interrupted")
                print(f"Saving emergency checkpoint to {emergency_path}...")
                self.history['nr_epochs'] = epochs_already_done + epoch + 1
                self.save(emergency_path)
        finally:
            end = time.time()
            elapsed = end - start

        # Update nr_epochs to reflect true total trained (SIGTERM path already set it)
        if not self._sigterm_received:
            self.history['nr_epochs'] = epochs_already_done + epochs_this_job

        print("elapsed:" + str(elapsed))

        # Get input/output variable names from dataset
        input_variables = [spec['name'] for spec in train_ds.get_input_spec()]
        output_variable = train_ds.get_output_spec()['name']

        if self.db:
            self.db.add_training_result(self.get_model_id(), "UNET", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss, testing_paths, test_loss, self.get_parameters(), self.spec.save())
        if model_path:
            self.save(model_path)

        # pass over the training and test sets and calculate model metrics
        metrics = {}
        metrics["test"] = self.evaluate(test_ds, device)
        metrics["train"] = self.evaluate(train_ds, device)

        self.dump_metrics("Test Metrics", metrics["test"])
        self.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.get_model_id(), training_paths, testing_paths, metrics)

    def summary(self):
        """
        Print a summary of the encoder/input and decoder/output layers
        """
        if self.architecture == 'flow_matching' and self.flow_model is not None:
            n_params = sum(p.numel() for p in self.flow_model.parameters())
            s = f"Flow Matching UNet Summary:\n"
            s += f"\tArchitecture: flow_matching\n"
            s += f"\tBase channels: {self.base_channels}\n"
            s += f"\tInference steps: {self.flow_steps}\n"
            s += f"\tTotal parameters: {n_params:,}\n"
            s += f"\tInput shape: {self.input_shape}\n"
            s += f"\tOutput shape: {self.output_shape}\n"
            return s
        elif self.architecture == 'standard' and self.encoder is not None:
            n_enc = sum(p.numel() for p in self.encoder.parameters())
            n_dec = sum(p.numel() for p in self.decoder.parameters())
            s = f"Standard Residual UNet Summary:\n"
            s += f"\tArchitecture: standard\n"
            s += f"\tBase channels: {self.base_channels}\n"
            s += f"\tEncoder parameters: {n_enc:,}\n"
            s += f"\tDecoder parameters: {n_dec:,}\n"
            s += f"\tTotal parameters: {n_enc + n_dec:,}\n"
            return s
        elif self.spec:
            s = "Model Summary:\n"
            for input_spec in self.spec.input_layers:
                s += str(input_spec)
            s += "\tFully Connected Layer:\n"
            s += f"\t\tsize={self.fc_size}\n"
            s += "\tLatent Vector:\n"
            s += f"\t\tsize={self.encoded_dim_size}\n"
            s += "\tFully Connected Layer:\n"
            s += f"\t\tsize={self.fc_size}\n"
            for output_spec in self.spec.output_layers:
                s += str(output_spec)
            return s
        else:
            return "Model has not been trained - no layers assigned yet"

    def save(self, to_folder):
        """
        Save the model to disk

        :param to_folder: folder to which model files are to be saved
        """
        os.makedirs(to_folder, exist_ok=True)

        if self.architecture == 'flow_matching':
            flow_model_path = os.path.join(to_folder, "flow_model.weights")
            torch.save(self.flow_model.state_dict(), flow_model_path)
        else:
            encoder_path = os.path.join(to_folder, "encoder.weights")
            torch.save(self.encoder.state_dict(), encoder_path)
            decoder_path = os.path.join(to_folder, "decoder.weights")
            torch.save(self.decoder.state_dict(), decoder_path)

        # Save optimizer state for proper training continuation
        if self.optim is not None:
            optimizer_path = os.path.join(to_folder, "optimizer.state")
            torch.save(self.optim.state_dict(), optimizer_path)

        # Save scheduler state if available
        if hasattr(self, '_scheduler') and self._scheduler is not None:
            scheduler_path = os.path.join(to_folder, "scheduler.state")
            torch.save(self._scheduler.state_dict(), scheduler_path)

        normalisation_path = os.path.join(to_folder, "normalisation.weights")
        with open(normalisation_path, "w") as f:
            f.write(json.dumps(self.normalisation_parameters))

        parameters = self.get_parameters()

        parameters_path = os.path.join(to_folder, "parameters.json")
        with open(parameters_path, "w") as f:
            f.write(json.dumps(parameters))

        spec_path = os.path.join(to_folder, "spec.json")
        with open(spec_path, "w") as f:
            f.write(json.dumps(self.spec.save()))

        history_path = os.path.join(to_folder, "history.json")
        with open(history_path, "w") as f:
            f.write(json.dumps(self.history))

        summary_path = os.path.join(to_folder, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(self.summary())
        super().save(to_folder)

    def load(self, from_folder):
        """
        Load a model from disk

        :param from_folder: folder from which model files should be loaded
        """
        normalisation_path = os.path.join(from_folder, "normalisation.weights")
        with open(normalisation_path, "r") as f:
            self.normalisation_parameters = json.loads(f.read())
        parameters_path = os.path.join(from_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())
            if "model_id" in parameters:
                self.set_model_id(parameters["model_id"])
            self.input_shape = tuple(parameters["input_shape"])
            self.output_shape = tuple(parameters["output_shape"])
            self.batch_size = parameters["batch_size"]
            self.test_interval = parameters["test_interval"]
            self.encoded_dim_size = parameters["encoded_dim_size"]
            self.fc_size = parameters["fc_size"]
            self.lr = parameters["lr"]
            self.weight_decay = parameters["weight_decay"]
            self.normalise_input = parameters["normalise_input"]
            self.normalise_output = parameters["normalise_output"]

            self.conv_kernel_size = parameters.get("conv_kernel_size", None)
            self.conv_stride = parameters.get("conv_stride", None)
            self.conv_input_layer_count = parameters.get("conv_input_layer_count", None)
            self.conv_output_layer_count = parameters.get("conv_output_layer_count", None)
            self.bottleneck_type = parameters.get("bottleneck_type", "fc")
            self.use_attention = parameters.get("use_attention", True)
            self.skip_mode = parameters.get("skip_mode", "concat")
            self.skip_dropout = parameters.get("skip_dropout", 0.0)
            self.skip_scale = parameters.get("skip_scale", 1.0)
            self.latent_activation = parameters.get("latent_activation", "relu")
            self.output_activation = parameters.get("output_activation", "sigmoid")
            self.predict_delta = parameters.get("predict_delta", False)
            self.delta_reference_channel = parameters.get("delta_reference_channel", None)
            self.architecture = parameters.get("architecture", "legacy")
            self.base_channels = parameters.get("base_channels", 64)
            self.flow_steps = parameters.get("flow_steps", 4)
            self.augment = parameters.get("augment", False)
            self.slope_direction_channel = parameters.get("slope_direction_channel", 7)
            self.lambda_pearson = parameters.get("lambda_pearson", 0)
            
        use_fc = (self.bottleneck_type == 'fc')

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        spec_path = os.path.join(from_folder, "spec.json")
        with open(spec_path) as f:
            self.spec = ModelSpec()
            self.spec.load(json.loads(f.read()))

        self.encoder = None
        self.decoder = None
        self.flow_model = None

        if self.architecture == 'flow_matching':
            input_chan = self.input_shape[0]
            output_chan = self.output_shape[0]
            self.flow_model = FlowMatchingUNet(
                cond_channels=input_chan, target_channels=output_chan,
                base_channels=self.base_channels, dropout_rate=self.dropout_rate
            )
            flow_model_path = os.path.join(from_folder, "flow_model.weights")
            self.flow_model.load_state_dict(self.torch_load(flow_model_path))
            self.flow_model.eval()
        elif self.architecture == 'standard':
            input_chan = self.input_shape[0]
            output_chan = self.output_shape[0]
            self.encoder = StandardEncoder(in_channels=input_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate)
            self.decoder = StandardDecoder(out_channels=output_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, output_activation=self.output_activation)
        else:
            self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate, use_fc=use_fc, latent_activation=self.latent_activation)
            self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate, use_fc=use_fc, use_attention=self.use_attention, skip_mode=self.skip_mode, skip_dropout=self.skip_dropout, skip_scale=self.skip_scale, latent_activation=self.latent_activation, output_activation=self.output_activation)

        if self.architecture != 'flow_matching':
            encoder_path = os.path.join(from_folder, "encoder.weights")
#         self.encoder.load_state_dict(torch.load(encoder_path))
            self.encoder.load_state_dict(self.torch_load(encoder_path))
            self.encoder.eval()
            decoder_path = os.path.join(from_folder, "decoder.weights")
#         self.decoder.load_state_dict(torch.load(decoder_path))
            self.decoder.load_state_dict(self.torch_load(decoder_path))
            self.decoder.eval()

        # Store optimizer/scheduler state paths for deferred restoration during training continuation
        # (Can't restore now because optimizer/scheduler don't exist yet — they're created in train_from_datasets)
        optimizer_path = os.path.join(from_folder, "optimizer.state")
        if os.path.exists(optimizer_path):
            self._saved_optimizer_state_path = optimizer_path
            print(f"Found saved optimizer state at {optimizer_path}")
        else:
            self._saved_optimizer_state_path = None

        scheduler_path = os.path.join(from_folder, "scheduler.state")
        if os.path.exists(scheduler_path):
            self._saved_scheduler_state_path = scheduler_path
            print(f"Found saved scheduler state at {scheduler_path}")
        else:
            self._saved_scheduler_state_path = None

        super().load(from_folder)

    def pearson_corr_torch(self, decoded_data, high_res):
        # flatten
        decoded_data_flat = decoded_data.view(decoded_data.size(0), decoded_data.size(1), -1)
        high_res_flat = high_res.view(high_res.size(0), high_res.size(1), -1)
        
        # compute the mean
        mean_decoded = torch.mean(decoded_data_flat, dim=2, keepdim=True)
        mean_high_res = torch.mean(high_res_flat, dim=2, keepdim=True)

        # subtracting the mean
        decoded_data_centered = decoded_data_flat - mean_decoded
        high_res_centered = high_res_flat - mean_high_res

        # compute standard deviations
        std_decoded = torch.std(decoded_data_centered, dim=2, keepdim=True)
        std_high_res = torch.std(high_res_centered, dim=2, keepdim=True)

        # normalize by dividing by the standard deviation
        decoded_data_normalized = decoded_data_centered / std_decoded
        high_res_normalized = high_res_centered / std_high_res

        # Pearson correlation
        correlation = torch.mean(decoded_data_normalized * high_res_normalized, dim=2)

        return correlation
    
    def tv_loss(self, x):
        """Calculate Total Variation Loss"""
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]        
