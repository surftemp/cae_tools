import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from cae_tools.models.standard_unet import StandardEncoder, StandardDecoder
from cae_tools.models.flow_matching_unet import (
    FlowMatchingUNet, flow_matching_loss, flow_matching_sample,
    ConditionedFlowMatchingUNet, conditioned_flow_matching_loss,
    conditioned_flow_matching_sample,
)
from cae_tools.models.conditioned_unet import ConditionedEncoder, ConditionedDecoder
from cae_tools.models.conditioned_helpers import unpack_batch, forward_pass, split_for_scoring
from torchvision import models
import torch.nn.functional as F


import numpy as np
import xarray as xr
import json
import os
import time
import signal
import sys

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


def compute_spectral_loss(pred, target):
    """
    Spectral loss: penalises mismatch in spatial frequency content.

    Computes 2D FFT of both pred and target (each of shape (B, 1, H, W)),
    takes the magnitude spectrum, applies log(1 + |F|) to compress dynamic
    range so high-frequency components are not swamped by low-frequency ones,
    then returns the mean squared difference between the two log-magnitude
    spectra.
    """
    fft_pred = torch.fft.fft2(pred)
    fft_target = torch.fft.fft2(target)
    log_mag_pred = torch.log1p(fft_pred.abs())
    log_mag_target = torch.log1p(fft_target.abs())
    return F.mse_loss(log_mag_pred, log_mag_target)


def compute_subgroup_robustness_loss(pred, target, era5_map, n_era5_bins=None,
                                     min_samples=50):
    """
    Variance of per-ERA5-bin MSEs.

    Quantile-bins pixels by ERA5 skin temperature, computes MSE within each
    bin, returns the variance across bins. Penalising this encourages uniform
    performance across temperature regimes.

    Args:
        pred: (B, 1, H, W) prediction
        target: (B, 1, H, W) target
        era5_map: (B, 1, H, W) ERA5 temperature map (normalised)
        n_era5_bins: number of quantile bins (default: batch size)
        min_samples: minimum pixels per bin to include

    Returns:
        (loss, info_dict)
    """
    if n_era5_bins is None:
        n_era5_bins = pred.shape[0]
    se = (pred - target).pow(2)
    era5_flat = era5_map.reshape(-1)
    se_flat = se.reshape(-1)

    quantiles = torch.linspace(0, 1, n_era5_bins + 1, device=era5_map.device)
    bin_edges = torch.quantile(era5_flat, quantiles)

    subgroup_mses = []
    for i in range(n_era5_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_era5_bins - 1:
            mask = (era5_flat >= lo) & (era5_flat < hi)
        else:
            mask = (era5_flat >= lo) & (era5_flat <= hi)
        n = mask.sum()
        if n >= min_samples:
            subgroup_mses.append(se_flat[mask].mean())

    if len(subgroup_mses) < 2:
        return torch.tensor(0.0, device=pred.device), {}

    subgroup_mses = torch.stack(subgroup_mses)
    robustness_loss = subgroup_mses.var()
    info = {f'mse_era5_bin_{i}': m.item() for i, m in enumerate(subgroup_mses)}
    info['n_subgroups'] = len(subgroup_mses)
    info['mse_var_across_bins'] = robustness_loss.item()
    return robustness_loss, info


def compute_local_variance_loss(pred, target, kernel_size=5):
    """
    Penalise mismatch in local spatial variance (position-tolerant texture matching).

    Computes variance in sliding windows for both pred and target, then
    penalises the difference. Says: "where the target has high texture, your
    prediction should have high texture too" — without requiring exact pixel
    alignment within each window.

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        kernel_size: window size for local variance computation
    """
    padding = kernel_size // 2
    ones = torch.ones(1, 1, kernel_size, kernel_size,
                      device=pred.device) / (kernel_size ** 2)

    # Local mean via uniform box filter
    pred_mean = F.conv2d(pred, ones, padding=padding)
    target_mean = F.conv2d(target, ones, padding=padding)

    # Local variance = E[x²] - E[x]²
    pred_var = F.conv2d(pred ** 2, ones, padding=padding) - pred_mean ** 2
    target_var = F.conv2d(target ** 2, ones, padding=padding) - target_mean ** 2

    return F.mse_loss(pred_var, target_var)


# ── Multi-scale Pearson loss ──────────────────────────────────────────────

def compute_multiscale_pearson_loss(pred, target, scales=(1, 2, 4, 8)):
    """
    Pearson correlation loss computed at multiple spatial scales.

    At each scale, avg_pool2d reduces spatial resolution, then Pearson R is
    computed. Finer scales are weighted higher (weight = 1/scale, normalised)
    so the model is forced to match patterns at neighbourhood and parcel
    level, not just large-scale gradients.

    Returns weighted mean of (1 - R) across scales.

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        scales: tuple of pool sizes (1 = full resolution)
    """
    weights = [1.0 / s for s in scales]
    total_weight = sum(weights)

    loss = torch.tensor(0.0, device=pred.device)

    for scale, w in zip(scales, weights):
        if scale == 1:
            p, t = pred, target
        else:
            p = F.avg_pool2d(pred, kernel_size=scale)
            t = F.avg_pool2d(target, kernel_size=scale)

        # Pearson R over spatial dims (B, 1, H', W') → scalar
        p_flat = p.reshape(p.shape[0], -1)  # (B, N)
        t_flat = t.reshape(t.shape[0], -1)

        p_mean = p_flat.mean(dim=1, keepdim=True)
        t_mean = t_flat.mean(dim=1, keepdim=True)

        p_c = p_flat - p_mean
        t_c = t_flat - t_mean

        cov = (p_c * t_c).sum(dim=1)
        std_p = (p_c ** 2).sum(dim=1).sqrt().clamp(min=1e-8)
        std_t = (t_c ** 2).sum(dim=1).sqrt().clamp(min=1e-8)

        r = cov / (std_p * std_t)
        scale_loss = (1 - r).mean()
        loss = loss + (w / total_weight) * scale_loss

    return loss


# ── Gradient (Sobel) loss ─────────────────────────────────────────────────

def _get_sobel_kernels(device):
    """Return Sobel kernels for horizontal and vertical gradients."""
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32, device=device)
    # Shape: (1, 1, 3, 3) for single-channel conv
    return sobel_x.reshape(1, 1, 3, 3), sobel_y.reshape(1, 1, 3, 3)


def compute_gradient_loss(pred, target):
    """
    Gradient loss using Sobel filters.

    Extracts horizontal and vertical gradients from both pred and target,
    then computes L1 loss between gradient maps. Directly penalises blurred
    edges at land cover boundaries, urban transitions, elevation breaks.

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
    """
    sobel_x, sobel_y = _get_sobel_kernels(pred.device)

    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    target_gx = F.conv2d(target, sobel_x, padding=1)
    target_gy = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


# ── Patch-wise SSIM loss ─────────────────────────────────────────────────

def _gaussian_kernel_1d(size, sigma, device):
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size, sigma, device):
    """Create 2D Gaussian kernel as outer product of two 1D kernels."""
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)
    return k2d.reshape(1, 1, size, size)


def compute_ssim_loss(pred, target, kernel_size=11, sigma=1.5,
                      data_range=None):
    """
    Patch-wise SSIM loss: 1 - mean(SSIM map).

    Computes SSIM in local Gaussian-weighted windows. Captures local
    luminance + contrast + structure. Supersedes local variance loss for
    fine-scale pattern matching.

    C1/C2 constants are scaled by data_range for output_activation='none'
    (unbounded output). If data_range is None, it is estimated from the
    target batch.

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        kernel_size: Gaussian window size (default 11)
        sigma: Gaussian sigma (default 1.5)
        data_range: dynamic range of the data. If None, uses
            max(target) - min(target) from the batch.
    """
    if data_range is None:
        data_range = (target.max() - target.min()).clamp(min=1e-8)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_2d(kernel_size, sigma, pred.device)
    pad = kernel_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad)
    mu_t = F.conv2d(target, kernel, padding=pad)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred ** 2, kernel, padding=pad) - mu_p_sq
    sigma_t_sq = F.conv2d(target ** 2, kernel, padding=pad) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad) - mu_pt

    numerator = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
    denominator = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)

    ssim_map = numerator / denominator
    return 1 - ssim_map.mean()


class UNET(BaseModel):
    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, dropout_rate=0.1, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None, lambda_l1=0.001, lambda_pearson=0,
                 checkpoint_interval=None, bottleneck_type='fc', use_attention=True,
                 skip_mode='concat', skip_dropout=0.0, skip_scale=1.0, latent_activation='relu',
                 output_activation='sigmoid', predict_delta=False, delta_reference_channel=None,
                 architecture='legacy', base_channels=64, flow_steps=4,
                 augment=False, slope_direction_channel=7, cond_dim=0,
                 lambda_spectral=0.0, spectral_every_k_epochs=10,
                 lambda_subgroup=0.0, lambda_cold=0.0,
                 cold_threshold_k=10.0, era5_channel_idx=3, era5_cond_idx=0,
                 lambda_local_var=0.0,
                 lambda_ms_pearson=0.0, lambda_gradient=0.0, lambda_ssim=0.0,
                 activation='relu',
                 n_res_blocks_hi=1,
                 cond_variables=None,
                 cond_inject_stages=None,
                 cond_method='concat',
                 lc_embed_dim=0):
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
        :param cond_inject_stages: set of stage names for conditioning injection (default: all stages)
        :param cond_method: conditioning method - 'concat' or 'film' (default: 'concat')
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
        self.cond_dim = cond_dim
        self.lambda_spectral = lambda_spectral
        self.spectral_every_k_epochs = spectral_every_k_epochs
        self.lambda_subgroup = lambda_subgroup
        self.lambda_cold = lambda_cold
        self.cold_threshold_k = cold_threshold_k
        self.era5_channel_idx = era5_channel_idx
        self.era5_cond_idx = era5_cond_idx
        self.lambda_local_var = lambda_local_var
        self.lambda_ms_pearson = lambda_ms_pearson
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim
        self.activation = activation
        self.n_res_blocks_hi = n_res_blocks_hi
        self.cond_variables = cond_variables or []
        self.input_variables = []
        self.cond_inject_stages = cond_inject_stages  # None = all stages (default)
        self.cond_method = cond_method
        self.lc_embed_dim = lc_embed_dim
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
            "cond_dim": self.cond_dim,
            "lambda_spectral": self.lambda_spectral,
            "spectral_every_k_epochs": self.spectral_every_k_epochs,
            "lambda_subgroup": self.lambda_subgroup,
            "lambda_cold": self.lambda_cold,
            "cold_threshold_k": self.cold_threshold_k,
            "era5_channel_idx": self.era5_channel_idx,
            "era5_cond_idx": self.era5_cond_idx,
            "lambda_local_var": self.lambda_local_var,
            "lambda_ms_pearson": self.lambda_ms_pearson,
            "lambda_gradient": self.lambda_gradient,
            "lambda_ssim": self.lambda_ssim,
            "activation": self.activation,
            "n_res_blocks_hi": self.n_res_blocks_hi,
            "cond_variables": self.cond_variables,
            "input_variables": self.input_variables,
            "cond_inject_stages": sorted(self.cond_inject_stages) if self.cond_inject_stages is not None else None,
            "cond_method": self.cond_method,
            "lc_embed_dim": self.lc_embed_dim,
            "model_id": self.get_model_id()
        }

    def get_input_variable_names(self):
        """Return input variable names needed by apply_cae.

        For conditioned models, returns spatial_vars + cond_vars so that
        apply_cae stacks all channels (broadcasting scalars to spatial dims).
        split_for_scoring then splits the last cond_dim channels back out.
        """
        base_names = super().get_input_variable_names()
        if self.architecture == 'conditioned' and self.cond_variables:
            if base_names is None:
                return None
            return base_names + self.cond_variables
        return base_names

    def _get_lc_embed_kwargs(self):
        """Build keyword args for land cover embedding in ConditionedEncoder.

        Returns a dict with lc_embed_dim, lc_num_classes, lc_min, lc_max,
        lc_channel_idx if embedding is enabled (lc_embed_dim > 0), else
        just lc_embed_dim=0.
        """
        if self.lc_embed_dim <= 0:
            return {'lc_embed_dim': 0}

        norm = self.normalisation_parameters
        if isinstance(norm, dict) and 'min_inputs' in norm:
            lc_min = norm['min_inputs'].get('land_cover', 0.0)
            lc_max = norm['max_inputs'].get('land_cover', 0.0)
        elif isinstance(norm, list) and len(norm) >= 2:
            lc_min = norm[0].get('land_cover', 0.0)
            lc_max = norm[1].get('land_cover', 0.0)
        else:
            raise ValueError(
                f"Cannot extract land_cover normalisation from parameters: {type(norm)}")

        num_classes = int(round(lc_max - lc_min)) + 1
        # Determine channel index
        if self.input_variables and 'land_cover' in self.input_variables:
            # For conditioned arch, spatial_variables are the non-cond subset
            spatial_vars = [v for v in self.input_variables
                            if v not in self.cond_variables]
            lc_idx = spatial_vars.index('land_cover') if 'land_cover' in spatial_vars else 0
        else:
            lc_idx = 0

        print(f"  LC embedding: dim={self.lc_embed_dim}, classes={num_classes}, "
              f"range=[{lc_min}, {lc_max}], channel_idx={lc_idx}")
        return {
            'lc_embed_dim': self.lc_embed_dim,
            'lc_num_classes': num_classes,
            'lc_min': lc_min,
            'lc_max': lc_max,
            'lc_channel_idx': lc_idx,
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

    def __train_epoch_from_loader(self, data_loader, device, global_epoch=0):
        """Train epoch iterating DataLoader directly (no batch preloading)."""
        self.encoder.train()
        self.decoder.train()
        lambda_pearson = self.lambda_pearson
        is_pattern_epoch = (self.spectral_every_k_epochs > 0 and
                            global_epoch % self.spectral_every_k_epochs == 0)
        acc = {k: 0.0 for k in ['mse', 'mae', 'pearson', 'spectral',
                                  'subgroup', 'cold', 'hard_cold', 'local_var',
                                  'ms_pearson', 'gradient', 'ssim']}
        n_batches = 0

        for i, (low_res, high_res, labels) in enumerate(data_loader):
            low_res = low_res.to(device, non_blocking=True)
            high_res = high_res.to(device, non_blocking=True)

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

            # Cold pixel loss
            if self.lambda_cold > 0:
                era5_map = self._extract_era5_map(inputs=low_res)
                l_cold = self.lambda_cold * self._soft_cold_pixel_rate(decoded_data, era5_map)
                combined_loss = combined_loss + l_cold
                acc['cold'] += l_cold.item()

            # Subgroup robustness loss
            if self.lambda_subgroup > 0:
                era5_map = self._extract_era5_map(inputs=low_res)
                l_sub, _ = compute_subgroup_robustness_loss(decoded_data, high_res, era5_map)
                combined_loss = combined_loss + self.lambda_subgroup * l_sub
                acc['subgroup'] += l_sub.item()

            # Spectral loss (K-th epochs only)
            if self.lambda_spectral > 0 and is_pattern_epoch:
                l_spec = self.lambda_spectral * compute_spectral_loss(decoded_data, high_res)
                combined_loss = combined_loss + l_spec
                acc['spectral'] += l_spec.item()

            # L1 loss
            if self.lambda_l1 > 0:
                l1_loss = (decoded_data - high_res).abs().mean()
                combined_loss = combined_loss + self.lambda_l1 * l1_loss

            # Local variance loss
            if self.lambda_local_var > 0:
                l_lv = self.lambda_local_var * compute_local_variance_loss(decoded_data, high_res)
                combined_loss = combined_loss + l_lv
                acc['local_var'] += l_lv.item()

            # Multi-scale Pearson loss
            if self.lambda_ms_pearson > 0:
                l_msp = self.lambda_ms_pearson * compute_multiscale_pearson_loss(decoded_data, high_res)
                combined_loss = combined_loss + l_msp
                acc['ms_pearson'] += l_msp.item()

            # Gradient (Sobel) loss
            if self.lambda_gradient > 0:
                l_grad = self.lambda_gradient * compute_gradient_loss(decoded_data, high_res)
                combined_loss = combined_loss + l_grad
                acc['gradient'] += l_grad.item()

            # SSIM loss
            if self.lambda_ssim > 0:
                l_ssim = self.lambda_ssim * compute_ssim_loss(decoded_data, high_res)
                combined_loss = combined_loss + l_ssim
                acc['ssim'] += l_ssim.item()

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)
            self.optim.step()

            acc['mse'] += mse_loss.item()
            acc['pearson'] += pearson_loss.item()
            with torch.no_grad():
                acc['mae'] += (decoded_data - high_res).abs().mean().item()
                if self.lambda_cold > 0 or self.lambda_subgroup > 0:
                    acc['hard_cold'] += self._hard_cold_pct(decoded_data, era5_map)
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in acc.items()}

    def __test_epoch_from_loader(self, data_loader, device, save_arr=None):
        """Test epoch iterating DataLoader directly (no batch preloading)."""
        acc = {k: 0.0 for k in ['mse', 'mae', 'pearson', 'hard_cold', 'spectral',
                                  'ms_pearson', 'gradient', 'ssim']}
        n_batches = 0
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ctr = 0
            for (low_res, high_res, labels) in data_loader:
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)

                acc['mse'] += self.loss_fn(decoded_data, high_res).item()
                acc['mae'] += (decoded_data - high_res).abs().mean().item()
                pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
                acc['pearson'] += (1 - torch.mean(pearson_corr)).item()
                era5_map = self._extract_era5_map(inputs=low_res)
                acc['hard_cold'] += self._hard_cold_pct(decoded_data, era5_map)
                acc['spectral'] += compute_spectral_loss(decoded_data, high_res).item()
                acc['ms_pearson'] += compute_multiscale_pearson_loss(decoded_data, high_res).item()
                acc['gradient'] += compute_gradient_loss(decoded_data, high_res).item()
                acc['ssim'] += compute_ssim_loss(decoded_data, high_res).item()

                if save_arr is not None:
                    B = decoded_data.shape[0]
                    save_arr[ctr:ctr + B, :, :, :] = decoded_data.cpu()
                ctr += decoded_data.shape[0]
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in acc.items()}

    def __train_epoch_flow_matching(self, data_loader, device, global_epoch=0):
        """Train epoch for flow matching: velocity loss + conventional auxiliary losses."""
        self.flow_model.train()
        is_conditioned_fm = (self.cond_dim > 0)
        is_pattern_epoch = (self.spectral_every_k_epochs > 0 and
                            global_epoch % self.spectral_every_k_epochs == 0)
        acc = {k: 0.0 for k in ['mse', 'mae', 'pearson', 'velocity_mse', 'spectral',
                                  'subgroup', 'cold', 'hard_cold', 'local_var',
                                  'ms_pearson', 'gradient', 'ssim']}
        n_batches = 0

        for batch in data_loader:
            if is_conditioned_fm:
                spatial_b, cond_b, target, _ = unpack_batch(batch, device, True)
                if self.augment:
                    spatial_b, target = augment_batch(
                        spatial_b, target,
                        slope_dir_channel=self.slope_direction_channel)
            else:
                conditioning, target, labels = batch
                conditioning = conditioning.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if self.augment:
                    conditioning, target = augment_batch(
                        conditioning, target,
                        slope_dir_channel=self.slope_direction_channel)

            self.optim.zero_grad()

            # Velocity loss + reconstructed prediction
            if is_conditioned_fm:
                velocity_loss, pred = conditioned_flow_matching_loss(
                    self.flow_model, spatial_b, cond_b, target, device=device)
            else:
                velocity_loss = flow_matching_loss(
                    self.flow_model, conditioning, target, device=device)
                pred = None  # no reconstruction for flat FM

            combined_loss = velocity_loss
            acc['velocity_mse'] += velocity_loss.item()

            # Conventional losses on reconstructed prediction (conditioned FM only)
            if pred is not None:
                mse_loss = self.loss_fn(pred, target)
                pearson_corr = self.pearson_corr_torch(pred, target)
                pearson_loss = 1 - torch.mean(pearson_corr)

                if self.lambda_pearson > 0:
                    combined_loss = combined_loss + self.lambda_pearson * pearson_loss

                if self.lambda_cold > 0:
                    era5_map = self._extract_era5_map(cond=cond_b)
                    l_cold = self.lambda_cold * self._soft_cold_pixel_rate(pred, era5_map)
                    combined_loss = combined_loss + l_cold
                    acc['cold'] += l_cold.item()

                if self.lambda_subgroup > 0:
                    era5_map = self._extract_era5_map(cond=cond_b)
                    l_sub, _ = compute_subgroup_robustness_loss(pred, target, era5_map)
                    combined_loss = combined_loss + self.lambda_subgroup * l_sub
                    acc['subgroup'] += l_sub.item()

                if self.lambda_spectral > 0 and is_pattern_epoch:
                    l_spec = self.lambda_spectral * compute_spectral_loss(pred, target)
                    combined_loss = combined_loss + l_spec
                    acc['spectral'] += l_spec.item()

                if self.lambda_l1 > 0:
                    l1_loss = (pred - target).abs().mean()
                    combined_loss = combined_loss + self.lambda_l1 * l1_loss

                if self.lambda_local_var > 0:
                    l_lv = self.lambda_local_var * compute_local_variance_loss(pred, target)
                    combined_loss = combined_loss + l_lv
                    acc['local_var'] += l_lv.item()

                if self.lambda_ms_pearson > 0:
                    l_msp = self.lambda_ms_pearson * compute_multiscale_pearson_loss(pred, target)
                    combined_loss = combined_loss + l_msp
                    acc['ms_pearson'] += l_msp.item()

                if self.lambda_gradient > 0:
                    l_grad = self.lambda_gradient * compute_gradient_loss(pred, target)
                    combined_loss = combined_loss + l_grad
                    acc['gradient'] += l_grad.item()

                if self.lambda_ssim > 0:
                    l_ssim = self.lambda_ssim * compute_ssim_loss(pred, target)
                    combined_loss = combined_loss + l_ssim
                    acc['ssim'] += l_ssim.item()

                acc['mse'] += mse_loss.item()
                acc['pearson'] += pearson_loss.item()
                with torch.no_grad():
                    acc['mae'] += (pred - target).abs().mean().item()
                    if self.lambda_cold > 0 or self.lambda_subgroup > 0:
                        acc['hard_cold'] += self._hard_cold_pct(pred, era5_map)
            else:
                acc['mse'] += velocity_loss.item()

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.optim.step()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in acc.items()}

    def __test_epoch_flow_matching(self, data_loader, device, save_arr=None):
        """Test epoch for flow matching: Euler integration + full conventional metrics."""
        self.flow_model.eval()
        is_conditioned_fm = (self.cond_dim > 0)
        acc = {k: 0.0 for k in ['mse', 'mae', 'pearson', 'velocity_mse', 'hard_cold',
                                  'spectral', 'ms_pearson', 'gradient', 'ssim']}
        n_batches = 0

        with torch.no_grad():
            ctr = 0
            for batch in data_loader:
                if is_conditioned_fm:
                    spatial_b, cond_b, target, _ = unpack_batch(batch, device, True)
                else:
                    conditioning, target, labels = batch
                    conditioning = conditioning.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                B = target.shape[0]

                # Velocity MSE (same as training, for comparison)
                t = torch.rand(B, device=device)
                noise = torch.randn_like(target)
                t_expand = t[:, None, None, None]
                x_t = (1.0 - t_expand) * noise + t_expand * target
                velocity_target = target - noise
                if is_conditioned_fm:
                    velocity_pred = self.flow_model(x_t, spatial_b, cond_b, t)
                else:
                    velocity_pred = self.flow_model(x_t, conditioning, t)
                acc['velocity_mse'] += F.mse_loss(velocity_pred, velocity_target).item()

                # Full Euler integration for prediction
                target_shape = (B, target.shape[1], target.shape[2], target.shape[3])
                if is_conditioned_fm:
                    predicted = conditioned_flow_matching_sample(
                        self.flow_model, spatial_b, cond_b, target_shape,
                        num_steps=self.flow_steps, device=device)
                else:
                    predicted = flow_matching_sample(
                        self.flow_model, conditioning, target_shape,
                        num_steps=self.flow_steps, device=device)

                # All conventional metrics
                acc['mse'] += self.loss_fn(predicted, target).item()
                acc['mae'] += (predicted - target).abs().mean().item()
                pearson_corr = self.pearson_corr_torch(predicted, target)
                acc['pearson'] += (1 - torch.mean(pearson_corr)).item()
                if is_conditioned_fm:
                    era5_map = self._extract_era5_map(cond=cond_b)
                else:
                    era5_map = self._extract_era5_map(inputs=conditioning)
                acc['hard_cold'] += self._hard_cold_pct(predicted, era5_map)
                acc['spectral'] += compute_spectral_loss(predicted, target).item()
                acc['ms_pearson'] += compute_multiscale_pearson_loss(predicted, target).item()
                acc['gradient'] += compute_gradient_loss(predicted, target).item()
                acc['ssim'] += compute_ssim_loss(predicted, target).item()

                if save_arr is not None:
                    save_arr[ctr:ctr + B, :, :, :] = predicted.cpu()
                ctr += B
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in acc.items()}

    def __train_epoch_conditioned(self, data_loader, device, global_epoch=0):
        """Train epoch for conditioned architecture (4-tuple batches)."""
        self.encoder.train()
        self.decoder.train()
        lambda_pearson = self.lambda_pearson
        is_pattern_epoch = (self.spectral_every_k_epochs > 0 and
                            global_epoch % self.spectral_every_k_epochs == 0)
        acc = {k: 0.0 for k in ['mse', 'mae', 'pearson', 'spectral',
                                  'subgroup', 'cold', 'hard_cold', 'local_var',
                                  'ms_pearson', 'gradient', 'ssim']}
        n_batches = 0

        for batch in data_loader:
            spatial_b, cond_b, targets_b, _ = unpack_batch(batch, device, True)

            if self.augment:
                spatial_b, targets_b = augment_batch(
                    spatial_b, targets_b,
                    slope_dir_channel=self.slope_direction_channel)

            self.optim.zero_grad()
            pred = forward_pass(self.encoder, self.decoder, spatial_b, cond_b)

            mse_loss = self.loss_fn(pred, targets_b)
            pearson_corr = self.pearson_corr_torch(pred, targets_b)
            pearson_loss = 1 - torch.mean(pearson_corr)

            combined_loss = mse_loss + lambda_pearson * pearson_loss

            # Cold pixel loss
            if self.lambda_cold > 0:
                era5_map = self._extract_era5_map(cond=cond_b)
                l_cold = self.lambda_cold * self._soft_cold_pixel_rate(pred, era5_map)
                combined_loss = combined_loss + l_cold
                acc['cold'] += l_cold.item()

            # Subgroup robustness loss
            if self.lambda_subgroup > 0:
                era5_map = self._extract_era5_map(cond=cond_b)
                l_sub, _ = compute_subgroup_robustness_loss(pred, targets_b, era5_map)
                combined_loss = combined_loss + self.lambda_subgroup * l_sub
                acc['subgroup'] += l_sub.item()

            # Spectral loss (K-th epochs only)
            if self.lambda_spectral > 0 and is_pattern_epoch:
                l_spec = self.lambda_spectral * compute_spectral_loss(pred, targets_b)
                combined_loss = combined_loss + l_spec
                acc['spectral'] += l_spec.item()

            # L1 loss
            if self.lambda_l1 > 0:
                l1_loss = (pred - targets_b).abs().mean()
                combined_loss = combined_loss + self.lambda_l1 * l1_loss

            # Local variance loss
            if self.lambda_local_var > 0:
                l_lv = self.lambda_local_var * compute_local_variance_loss(pred, targets_b)
                combined_loss = combined_loss + l_lv
                acc['local_var'] += l_lv.item()

            # Multi-scale Pearson loss
            if self.lambda_ms_pearson > 0:
                l_msp = self.lambda_ms_pearson * compute_multiscale_pearson_loss(pred, targets_b)
                combined_loss = combined_loss + l_msp
                acc['ms_pearson'] += l_msp.item()

            # Gradient (Sobel) loss
            if self.lambda_gradient > 0:
                l_grad = self.lambda_gradient * compute_gradient_loss(pred, targets_b)
                combined_loss = combined_loss + l_grad
                acc['gradient'] += l_grad.item()

            # SSIM loss
            if self.lambda_ssim > 0:
                l_ssim = self.lambda_ssim * compute_ssim_loss(pred, targets_b)
                combined_loss = combined_loss + l_ssim
                acc['ssim'] += l_ssim.item()

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0)
            self.optim.step()

            acc['mse'] += mse_loss.item()
            acc['pearson'] += pearson_loss.item()
            with torch.no_grad():
                acc['mae'] += (pred - targets_b).abs().mean().item()
                if self.lambda_cold > 0 or self.lambda_subgroup > 0:
                    acc['hard_cold'] += self._hard_cold_pct(pred, era5_map)
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in acc.items()}

    def __test_epoch_conditioned(self, data_loader, device, save_arr=None):
        """Test epoch for conditioned architecture (4-tuple batches)."""
        acc = {k: 0.0 for k in ['mse', 'mae', 'pearson', 'hard_cold', 'spectral',
                                  'ms_pearson', 'gradient', 'ssim']}
        n_batches = 0
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ctr = 0
            for batch in data_loader:
                spatial_b, cond_b, targets_b, _ = unpack_batch(batch, device, True)
                pred = forward_pass(self.encoder, self.decoder, spatial_b, cond_b)

                acc['mse'] += self.loss_fn(pred, targets_b).item()
                acc['mae'] += (pred - targets_b).abs().mean().item()
                pearson_corr = self.pearson_corr_torch(pred, targets_b)
                acc['pearson'] += (1 - torch.mean(pearson_corr)).item()
                era5_map = self._extract_era5_map(cond=cond_b)
                acc['hard_cold'] += self._hard_cold_pct(pred, era5_map)
                acc['spectral'] += compute_spectral_loss(pred, targets_b).item()
                acc['ms_pearson'] += compute_multiscale_pearson_loss(pred, targets_b).item()
                acc['gradient'] += compute_gradient_loss(pred, targets_b).item()
                acc['ssim'] += compute_ssim_loss(pred, targets_b).item()

                if save_arr is not None:
                    B = pred.shape[0]
                    save_arr[ctr:ctr + B, :, :, :] = pred.cpu()
                ctr += pred.shape[0]
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in acc.items()}

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
                if self.cond_dim > 0:
                    # Conditioned FM: split input into spatial + cond scalars
                    spatial_ch = self.input_shape[0] - self.cond_dim
                    cond_indices = list(range(spatial_ch, self.input_shape[0]))
                    for input_data in batches:
                        spatial_b, cond_b = split_for_scoring(input_data, cond_indices)
                        B = spatial_b.shape[0]
                        target_shape = (B, self.output_shape[0], self.output_shape[1], self.output_shape[2])
                        predicted = conditioned_flow_matching_sample(
                            self.flow_model, spatial_b, cond_b, target_shape,
                            num_steps=self.flow_steps, device=device)
                        save_arr[ctr:ctr + B, :, :, :] = predicted.cpu()
                        ctr += B
                else:
                    for conditioning in batches:
                        B = conditioning.shape[0]
                        target_shape = (B, self.output_shape[0], self.output_shape[1], self.output_shape[2])
                        predicted = flow_matching_sample(
                            self.flow_model, conditioning, target_shape,
                            num_steps=self.flow_steps, device=device)
                        save_arr[ctr:ctr + B, :, :, :] = predicted.cpu()
                        ctr += B
        elif self.architecture == 'conditioned':
            self.encoder.eval()
            self.decoder.eval()
            device = next(self.encoder.parameters()).device
            # cond channels are the last self.cond_dim channels
            spatial_ch = self.input_shape[0] - self.cond_dim
            cond_indices = list(range(spatial_ch, self.input_shape[0]))
            with torch.no_grad():
                ctr = 0
                for input_data in batches:
                    spatial_b, cond_b = split_for_scoring(input_data, cond_indices)
                    predicted = forward_pass(self.encoder, self.decoder, spatial_b, cond_b)
                    B = predicted.shape[0]
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
                self.encoder = StandardEncoder(in_channels=input_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, activation=self.activation)
            else:
                self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate, use_fc=use_fc, latent_activation=self.latent_activation)
        if not self.decoder:
            if self.architecture == 'standard':
                self.decoder = StandardDecoder(out_channels=output_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, output_activation=self.output_activation, activation=self.activation)
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

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True)

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
                if self.cond_dim > 0:
                    # Conditioned flow matching: separate spatial/scalar inputs
                    spatial_ch = input_chan  # from get_input_shape() = spatial only
                    self.input_shape = (spatial_ch + self.cond_dim, input_y, input_x)
                    if not self.cond_variables and hasattr(train_ds, 'get_cond_variables'):
                        self.cond_variables = train_ds.get_cond_variables()
                    if not self.input_variables:
                        self.input_variables = list(train_ds.input_variables)
                    _inject = set(self.cond_inject_stages) if self.cond_inject_stages is not None else None
                    self.flow_model = ConditionedFlowMatchingUNet(
                        spatial_channels=spatial_ch, target_channels=output_chan,
                        cond_dim=self.cond_dim, base_channels=self.base_channels,
                        dropout_rate=self.dropout_rate, activation=self.activation,
                        inject_stages=_inject, cond_method=self.cond_method)
                    print(f"Conditioned Flow Matching UNet: spatial_ch={spatial_ch}, cond_dim={self.cond_dim}")
                    _inject_str = ','.join(sorted(_inject)) if _inject else 'all'
                    print(f"  ERA5 injection: stages={_inject_str}, method={self.cond_method}")
                else:
                    self.flow_model = FlowMatchingUNet(
                        cond_channels=input_chan, target_channels=output_chan,
                        base_channels=self.base_channels, dropout_rate=self.dropout_rate)
                print(f"Flow Matching UNet: {sum(p.numel() for p in self.flow_model.parameters()):,} parameters")
                print(f"Inference steps: {self.flow_steps}")
        elif self.architecture == 'conditioned':
            # get_input_shape() returns spatial channels only for conditioned datasets.
            # Store total (spatial + cond) as input_shape so apply_cae and load() work.
            spatial_ch = input_chan  # from get_input_shape() = spatial only
            
            if not self.cond_variables and hasattr(train_ds, 'get_cond_variables'):
                self.cond_variables = train_ds.get_cond_variables()
            if not self.input_variables:
                self.input_variables = list(train_ds.input_variables)
                
            self.input_shape = (spatial_ch + self.cond_dim, input_y, input_x)
            # Convert cond_inject_stages list to set for encoder/decoder
            _inject = set(self.cond_inject_stages) if self.cond_inject_stages is not None else None
            _lc_kw = self._get_lc_embed_kwargs()
            if not self.encoder:
                self.encoder = ConditionedEncoder(
                    spatial_in_channels=spatial_ch, cond_dim=self.cond_dim,
                    base_channels=self.base_channels, dropout_rate=self.dropout_rate,
                    activation=self.activation, n_res_blocks_hi=self.n_res_blocks_hi,
                    inject_stages=_inject, cond_method=self.cond_method,
                    **_lc_kw)
            if not self.decoder:
                self.decoder = ConditionedDecoder(
                    out_channels=output_chan, cond_dim=self.cond_dim,
                    base_channels=self.base_channels, dropout_rate=self.dropout_rate,
                    output_activation=self.output_activation, activation=self.activation,
                    n_res_blocks_hi=self.n_res_blocks_hi,
                    inject_stages=_inject, cond_method=self.cond_method)
            n_enc = sum(p.numel() for p in self.encoder.parameters())
            n_dec = sum(p.numel() for p in self.decoder.parameters())
            print(f"Conditioned UNet: spatial_ch={spatial_ch}, cond_dim={self.cond_dim}")
            print(f"  Encoder: {n_enc:,}, Decoder: {n_dec:,}, Total: {n_enc+n_dec:,}")
        else:
            if not self.encoder:
                if self.architecture == 'standard':
                    self.encoder = StandardEncoder(in_channels=input_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, activation=self.activation)
                else:
                    self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, dropout_rate=self.dropout_rate, use_fc=use_fc, latent_activation=self.latent_activation)
            if not self.decoder:
                if self.architecture == 'standard':
                    self.decoder = StandardDecoder(out_channels=output_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, output_activation=self.output_activation, activation=self.activation)
                else:
                    self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, dropout_rate=self.dropout_rate, use_fc=use_fc, use_attention=self.use_attention, skip_mode=self.skip_mode, skip_dropout=self.skip_dropout, skip_scale=self.skip_scale, latent_activation=self.latent_activation, output_activation=self.output_activation)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        print(f'Running on device: {device}')
        if self.activation != 'relu':
            print(f'Activation: {self.activation}')

        start = time.time()

        # Save training command for reproducibility
        if model_path:
            os.makedirs(model_path, exist_ok=True)
            cmd_path = os.path.join(model_path, 'training_command.txt')
            with open(cmd_path, 'w') as f:
                f.write(' '.join(sys.argv))

        self.loss_fn = torch.nn.MSELoss()

        _any_aux = (self.lambda_spectral > 0 or self.lambda_subgroup > 0 or
                    self.lambda_cold > 0 or self.lambda_local_var > 0 or
                    self.lambda_ms_pearson > 0 or self.lambda_gradient > 0 or
                    self.lambda_ssim > 0)
        if _any_aux:
            print(f'Aux losses: lambda_spectral={self.lambda_spectral}, '
                  f'spectral_every_k={self.spectral_every_k_epochs}, '
                  f'lambda_subgroup={self.lambda_subgroup}, '
                  f'lambda_cold={self.lambda_cold}, '
                  f'cold_threshold_k={self.cold_threshold_k}, '
                  f'lambda_l1={self.lambda_l1}, '
                  f'lambda_local_var={self.lambda_local_var}')
            print(f'Spatial losses: lambda_ms_pearson={self.lambda_ms_pearson}, '
                  f'lambda_gradient={self.lambda_gradient}, '
                  f'lambda_ssim={self.lambda_ssim}')

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

        # ---- Best-checkpoint tracking ----
        EMA_ALPHA = 0.3
        _ema_test = self.history.get('_ema_test_mse', None)
        _best_ema_test = self.history.get('_best_ema_test_mse', float('inf'))
        _best_test_mse = self.history.get('_best_test_mse', float('inf'))
        _ema_ratio = self.history.get('_ema_ratio', None)
        _best_ema_ratio = self.history.get('_best_train_test_ratio', float('inf'))
        RATIO_WARMUP_EPOCHS = 50

        try:
            for epoch in range(epochs_this_job):
                epoch_start = time.time()
                global_epoch = epochs_already_done + epoch

                # ---- Train epoch ----
                if self.architecture == 'flow_matching':
                    train_m = self.__train_epoch_flow_matching(train_loader, device, global_epoch)
                elif self.architecture == 'conditioned':
                    train_m = self.__train_epoch_conditioned(train_loader, device, global_epoch)
                else:
                    train_m = self.__train_epoch_from_loader(train_loader, device, global_epoch)
                train_loss = train_m['mse']

                if global_epoch < T_max:
                    scheduler.step()

                # ---- Test epoch + logging ----
                if epoch % self.test_interval == 0:
                    if self.architecture == 'flow_matching':
                        test_m = self.__test_epoch_flow_matching(test_loader, device)
                    elif self.architecture == 'conditioned':
                        test_m = self.__test_epoch_conditioned(test_loader, device)
                    else:
                        test_m = self.__test_epoch_from_loader(test_loader, device)

                    test_loss = test_m['mse']
                    test_mae = test_m.get('mae', 0.0)
                    test_pearson = test_m.get('pearson', 0.0)
                    train_mae = train_m.get('mae', 0.0)
                    hard_cold_test = test_m.get('hard_cold', 0.0)
                    hard_cold_train = train_m.get('hard_cold', 0.0)
                    lr = self.get_lr(self.optim)

                    # EMA tracking
                    if _ema_test is None:
                        _ema_test = test_loss
                    else:
                        _ema_test = EMA_ALPHA * test_loss + (1 - EMA_ALPHA) * _ema_test
                    ratio = test_loss / train_loss if train_loss > 0 else float('inf')
                    if _ema_ratio is None:
                        _ema_ratio = ratio
                    else:
                        _ema_ratio = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * _ema_ratio

                    # ---- Logging ----
                    is_fm = (self.architecture == 'flow_matching')
                    is_conditioned_fm = is_fm and self.cond_dim > 0
                    print(f"\nepoch: {global_epoch}")
                    print(f"  l_mse:      train={train_loss:.6f}  test={test_loss:.6f}  "
                          f"ema_test={_ema_test:.6f}  ema_ratio={_ema_ratio:.3f}")
                    if is_fm:
                        print(f"  velocity:   train={train_m.get('velocity_mse',0):.6f}  "
                              f"test={test_m.get('velocity_mse',0):.6f}")
                    if is_conditioned_fm or not is_fm:
                        print(f"  metrics:    train_rmse={train_loss**0.5:.6f}  "
                              f"test_rmse={test_loss**0.5:.6f}  "
                              f"train_mae={train_mae:.6f}  test_mae={test_mae:.6f}  "
                              f"rmse/mae={test_loss**0.5/max(test_mae,1e-10):.3f}")
                        try:
                            out_range = self._get_output_range_k()
                            print(f"  physical:   train_rmse={train_loss**0.5*out_range:.2f}K  "
                                  f"test_rmse={test_loss**0.5*out_range:.2f}K  "
                                  f"train_mae={train_mae*out_range:.2f}K  "
                                  f"test_mae={test_mae*out_range:.2f}K")
                        except Exception:
                            pass
                        print(f"  pearson:    train={train_m.get('pearson',0):.4f}  "
                              f"test={test_pearson:.4f}")
                    else:
                        # Flat FM (no conditioned data) — limited metrics
                        print(f"  metrics:    test_rmse={test_loss**0.5:.6f}  "
                              f"test_mae={test_mae:.6f}  "
                              f"rmse/mae={test_loss**0.5/max(test_mae,1e-10):.3f}")
                        try:
                            out_range = self._get_output_range_k()
                            print(f"  physical:   test_rmse={test_loss**0.5*out_range:.2f}K  "
                                  f"test_mae={test_mae*out_range:.2f}K")
                        except Exception:
                            pass
                    if hard_cold_test > 0 or hard_cold_train > 0:
                        print(f"  cold%:      train={hard_cold_train:.4f}%  "
                              f"test={hard_cold_test:.4f}%")
                    print(f"  spectral:   train={train_m.get('spectral',0):.6f}  "
                          f"test={test_m.get('spectral',0):.6f}")
                    if train_m.get('subgroup', 0) > 0 or train_m.get('cold', 0) > 0 or train_m.get('local_var', 0) > 0:
                        print(f"  aux:        subgroup={train_m.get('subgroup',0):.6f}  "
                              f"cold={train_m.get('cold',0):.6f}  "
                              f"local_var={train_m.get('local_var',0):.6f}")
                    if train_m.get('ms_pearson', 0) > 0 or train_m.get('gradient', 0) > 0 or train_m.get('ssim', 0) > 0:
                        print(f"  spatial:    ms_pearson={train_m.get('ms_pearson',0):.6f}  "
                              f"gradient={train_m.get('gradient',0):.6f}  "
                              f"ssim={train_m.get('ssim',0):.6f}")
                        print(f"  spatial(t): ms_pearson={test_m.get('ms_pearson',0):.6f}  "
                              f"gradient={test_m.get('gradient',0):.6f}  "
                              f"ssim={test_m.get('ssim',0):.6f}")
                    print(f"  lr:         {lr:.2e}")

                    # ---- History ----
                    h = self.history
                    h.setdefault('train_loss', []).append(float(train_loss))
                    h.setdefault('test_loss', []).append(float(test_loss))
                    h.setdefault('test_pearson_loss', []).append(float(test_pearson))
                    h.setdefault('test_mae', []).append(float(test_mae))
                    h.setdefault('train_mae', []).append(float(train_mae))
                    h.setdefault('hard_cold_test_pct', []).append(float(hard_cold_test))
                    h.setdefault('hard_cold_train_pct', []).append(float(hard_cold_train))
                    h.setdefault('train_spectral', []).append(float(train_m.get('spectral', 0)))
                    h.setdefault('test_spectral', []).append(float(test_m.get('spectral', 0)))                    
                    h['_ema_test_mse'] = float(_ema_test)
                    h['_ema_ratio'] = float(_ema_ratio)

                    # ---- Checkpoint: best raw test MSE (primary) ----
                    if model_path and test_loss < _best_test_mse:
                        _best_test_mse = test_loss
                        h['_best_test_mse'] = float(_best_test_mse)
                        h['_best_test_mse_epoch'] = global_epoch
                        h['nr_epochs'] = global_epoch + 1
                        best_path = os.path.join(model_path, "checkpoint_best_test_mse")
                        print(f"  ★ New best test MSE {test_loss:.6f} at epoch {global_epoch} → {best_path}")
                        self.save(best_path)

                    # ---- Checkpoint: best EMA test MSE (smoothed) ----
                    if model_path and _ema_test < _best_ema_test:
                        _best_ema_test = _ema_test
                        h['_best_ema_test_mse'] = float(_best_ema_test)
                        h['_best_ema_test_epoch'] = global_epoch

                    # ---- Checkpoint: best ratio ----
                    ratio_ready = global_epoch >= RATIO_WARMUP_EPOCHS and _ema_ratio > 1.0
                    if model_path and ratio_ready and _ema_ratio < _best_ema_ratio:
                        _best_ema_ratio = _ema_ratio
                        h['_best_train_test_ratio'] = float(_best_ema_ratio)
                        h['_best_ratio_epoch'] = global_epoch
                        h['nr_epochs'] = global_epoch + 1
                        best_ratio_path = os.path.join(model_path, "checkpoint_best_ratio")
                        print(f"  ★ New best EMA ratio {_ema_ratio:.3f} at epoch {global_epoch} → {best_ratio_path}")
                        self.save(best_ratio_path)

                # Periodic checkpoint
                if self.checkpoint_interval and model_path and (global_epoch + 1) % self.checkpoint_interval == 0:
                    self.history['nr_epochs'] = global_epoch + 1
                    checkpoint_path = os.path.join(model_path, f"checkpoint_epoch_{global_epoch + 1}")
                    print(f"Saving checkpoint to {checkpoint_path}...")
                    self.save(checkpoint_path)

                # SIGTERM
                if self._sigterm_received:
                    print(f"[SIGTERM] Saving at global epoch {global_epoch + 1}...")
                    self.history['nr_epochs'] = global_epoch + 1
                    if model_path:
                        emergency_path = os.path.join(model_path, f"checkpoint_epoch_{global_epoch + 1}_sigterm")
                        self.save(emergency_path)
                        self.save(model_path)
                    break

                # Epoch timing
                epoch_sec = time.time() - epoch_start
                if epoch % self.test_interval == 0:
                    print(f"  time:       {epoch_sec:.1f}s  ({3600/max(epoch_sec,0.1):.1f} epochs/hr)")

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

        if not self._sigterm_received:
            self.history['nr_epochs'] = epochs_already_done + epochs_this_job

        print(f"Elapsed: {elapsed:.1f}s")

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
            if self.cond_dim > 0:
                spatial_ch = self.input_shape[0] - self.cond_dim
                s += f"\tSpatial channels: {spatial_ch}\n"
                s += f"\tConditioning dim: {self.cond_dim}\n"
                _inject = getattr(self, 'cond_inject_stages', None)
                _inject_str = ','.join(sorted(_inject)) if _inject else 'all'
                s += f"\tERA5 injection: stages={_inject_str}, method={self.cond_method}\n"
            s += f"\tTotal parameters: {n_params:,}\n"
            s += f"\tInput shape: {self.input_shape}\n"
            s += f"\tOutput shape: {self.output_shape}\n"
            return s
        elif self.architecture == 'conditioned' and self.encoder is not None:
            n_enc = sum(p.numel() for p in self.encoder.parameters())
            n_dec = sum(p.numel() for p in self.decoder.parameters())
            spatial_ch = self.input_shape[0] - self.cond_dim
            s = f"Conditioned UNet Summary:\n"
            s += f"\tArchitecture: conditioned\n"
            s += f"\tBase channels: {self.base_channels}\n"
            s += f"\tSpatial channels: {spatial_ch}\n"
            s += f"\tConditioning dim: {self.cond_dim}\n"
            s += f"\tEncoder parameters: {n_enc:,}\n"
            s += f"\tDecoder parameters: {n_dec:,}\n"
            s += f"\tTotal parameters: {n_enc + n_dec:,}\n"
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
            self.cond_dim = parameters.get("cond_dim", 0)
            self.lambda_spectral = parameters.get("lambda_spectral", 0.0)
            self.spectral_every_k_epochs = parameters.get("spectral_every_k_epochs", 10)
            self.lambda_subgroup = parameters.get("lambda_subgroup", 0.0)
            self.lambda_cold = parameters.get("lambda_cold", 0.0)
            self.cold_threshold_k = parameters.get("cold_threshold_k", 10.0)
            self.era5_channel_idx = parameters.get("era5_channel_idx", 3)
            self.era5_cond_idx = parameters.get("era5_cond_idx", 0)
            self.lambda_local_var = parameters.get("lambda_local_var", 0.0)
            self.lambda_ms_pearson = parameters.get("lambda_ms_pearson", 0.0)
            self.lambda_gradient = parameters.get("lambda_gradient", 0.0)
            self.lambda_ssim = parameters.get("lambda_ssim", 0.0)
            self.activation = parameters.get("activation", "relu")
            self.n_res_blocks_hi = parameters.get("n_res_blocks_hi", 1)
            self.cond_variables = parameters.get("cond_variables", [])
            self.input_variables = parameters.get("input_variables", [])
            _stages = parameters.get("cond_inject_stages", None)
            self.cond_inject_stages = _stages  # None = all stages (default)
            self.cond_method = parameters.get("cond_method", "concat")
            self.lc_embed_dim = parameters.get("lc_embed_dim", 0)

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
            if self.cond_dim > 0:
                spatial_ch = input_chan - self.cond_dim
                _inject = set(self.cond_inject_stages) if self.cond_inject_stages is not None else None
                self.flow_model = ConditionedFlowMatchingUNet(
                    spatial_channels=spatial_ch, target_channels=output_chan,
                    cond_dim=self.cond_dim, base_channels=self.base_channels,
                    dropout_rate=self.dropout_rate, activation=self.activation,
                    inject_stages=_inject, cond_method=self.cond_method)
            else:
                self.flow_model = FlowMatchingUNet(
                    cond_channels=input_chan, target_channels=output_chan,
                    base_channels=self.base_channels, dropout_rate=self.dropout_rate)
            flow_model_path = os.path.join(from_folder, "flow_model.weights")
            self.flow_model.load_state_dict(self.torch_load(flow_model_path))
            self.flow_model.eval()
        elif self.architecture == 'conditioned':
            input_chan = self.input_shape[0]
            output_chan = self.output_shape[0]
            spatial_ch = input_chan - self.cond_dim
            _inject = set(self.cond_inject_stages) if self.cond_inject_stages is not None else None
            _lc_kw = self._get_lc_embed_kwargs()
            self.encoder = ConditionedEncoder(
                spatial_in_channels=spatial_ch, cond_dim=self.cond_dim,
                base_channels=self.base_channels, dropout_rate=self.dropout_rate,
                activation=self.activation, n_res_blocks_hi=self.n_res_blocks_hi,
                inject_stages=_inject, cond_method=self.cond_method,
                **_lc_kw)
            self.decoder = ConditionedDecoder(
                out_channels=output_chan, cond_dim=self.cond_dim,
                base_channels=self.base_channels, dropout_rate=self.dropout_rate,
                output_activation=self.output_activation, activation=self.activation,
                n_res_blocks_hi=self.n_res_blocks_hi,
                inject_stages=_inject, cond_method=self.cond_method)
        elif self.architecture == 'standard':
            input_chan = self.input_shape[0]
            output_chan = self.output_shape[0]
            self.encoder = StandardEncoder(in_channels=input_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, activation=self.activation)
            self.decoder = StandardDecoder(out_channels=output_chan, base_channels=self.base_channels, dropout_rate=self.dropout_rate, output_activation=self.output_activation, activation=self.activation)
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


    def _get_output_range_k(self):
        """Return (max_output - min_output) in Kelvin for denormalisation."""
        norm_params = self.normalisation_parameters
        if isinstance(norm_params, list):
            # List-format normalisation: [min_inputs, max_inputs, min_outputs, max_outputs]
            min_out = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
        else:
            min_out = norm_params['min_output']
            max_out = norm_params['max_output']
        return max_out - min_out

    def _get_era5_norm_range(self):
        """Return (min, max) of ERA5 normalised range for cold pixel calc."""
        norm_params = self.normalisation_parameters
        if isinstance(norm_params, list):
            era5_key = 'era5_skt'
            min_in = norm_params[0].get(era5_key, 0.0)
            max_in = norm_params[1].get(era5_key, 1.0)
        else:
            min_in = 0.0
            max_in = 1.0
        return min_in, max_in

    def _soft_cold_pixel_rate(self, pred, era5_map):
        """
        Soft cold pixel rate: fraction of pixels predicted colder than
        ERA5 - cold_threshold_k (in Kelvin), using a smooth sigmoid.

        Args:
            pred: (B, 1, H, W) normalised prediction
            era5_map: (B, 1, H, W) normalised ERA5 skin temperature
        """
        norm_params = self.normalisation_parameters
        if isinstance(norm_params, list):
            min_out = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
            era5_min, era5_max = self._get_era5_norm_range()
        else:
            min_out = norm_params.get('min_output', 0.0)
            max_out = norm_params.get('max_output', 1.0)
            era5_min, era5_max = 0.0, 1.0
        pred_k = pred * (max_out - min_out) + min_out
        era5_k = era5_map * (era5_max - era5_min) + era5_min
        threshold_k = era5_k - self.cold_threshold_k
        soft_cold = torch.sigmoid(2.0 * (threshold_k - pred_k))
        return soft_cold.mean()

    def _hard_cold_pct(self, pred, era5_map):
        """
        Hard cold pixel percentage: fraction of pixels predicted colder than
        ERA5 - cold_threshold_k (in Kelvin).

        Args:
            pred: (B, 1, H, W) normalised prediction
            era5_map: (B, 1, H, W) normalised ERA5 skin temperature
        """
        norm_params = self.normalisation_parameters
        if isinstance(norm_params, list):
            min_out = norm_params[2].get('ST_slices', list(norm_params[2].values())[0])
            max_out = norm_params[3].get('ST_slices', list(norm_params[3].values())[0])
            era5_min, era5_max = self._get_era5_norm_range()
        else:
            min_out = norm_params.get('min_output', 0.0)
            max_out = norm_params.get('max_output', 1.0)
            era5_min, era5_max = 0.0, 1.0
        pred_k = pred * (max_out - min_out) + min_out
        era5_k = era5_map * (era5_max - era5_min) + era5_min
        threshold_k = era5_k - self.cold_threshold_k
        cold_mask = pred_k < threshold_k
        return 100.0 * cold_mask.float().mean().item()

    def _extract_era5_map(self, inputs=None, cond=None):
        """
        Extract ERA5 skin temperature as (B, 1, H, W) from inputs or cond.

        If cond is provided: era5 is conditioning scalar era5_cond_idx, broadcast to spatial.
        Otherwise: era5 is spatial channel era5_channel_idx from inputs.
        """
        if cond is not None:
            B = cond.shape[0]
            era5_scalar = cond[:, self.era5_cond_idx:self.era5_cond_idx + 1]  # (B, 1)
            H, W = self.output_shape[1], self.output_shape[2]
            return era5_scalar.unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)
        else:
            return inputs[:, self.era5_channel_idx:self.era5_channel_idx + 1, :, :]

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
        eps = 1e-8  # safe-guard 
        decoded_data_normalized = decoded_data_centered / (std_decoded + eps)
        high_res_normalized = high_res_centered / (std_high_res + eps)

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
