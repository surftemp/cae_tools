"""
Conditioned UNet Training Helpers.

Drop-in helper functions for integrating conditioned architecture into the
existing UNET.train_from_datasets() training loop. These are designed to be
called from within the existing method, minimising changes to the main loop.

Usage in train_from_datasets():

    from cae_tools.models.conditioned_helpers import (
        unpack_batch, forward_pass, create_conditioned_encoder_decoder)
"""

import torch


def create_conditioned_encoder_decoder(spatial_in_channels, cond_dim, output_channels,
                                        base_channels=64, dropout_rate=0.0,
                                        output_activation='none'):
    """
    Create ConditionedEncoder and ConditionedDecoder pair.

    Args:
        spatial_in_channels: number of spatial input channels (e.g. 8)
        cond_dim: conditioning vector dimension (e.g. 11 scalars or 44 for 2×2)
        output_channels: number of output channels (1 for temperature)
        base_channels: UNet base channel count
        dropout_rate: dropout rate for ResidualBlocks
        output_activation: 'none', 'sigmoid', or 'tanh'

    Returns:
        (encoder, decoder) tuple
    """
    from cae_tools.models.conditioned_unet import (
        ConditionedEncoder, ConditionedDecoder)

    encoder = ConditionedEncoder(
        spatial_in_channels=spatial_in_channels,
        cond_dim=cond_dim,
        base_channels=base_channels,
        dropout_rate=dropout_rate)

    decoder = ConditionedDecoder(
        out_channels=output_channels,
        cond_dim=cond_dim,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
        output_activation=output_activation)

    return encoder, decoder


def unpack_batch(batch, device, is_conditioned):
    """
    Unpack a batch from either standard or conditioned dataset.

    Standard dataset returns:  (inputs, targets, labels)   — 3-tuple
    Conditioned dataset returns: (spatial, cond, targets, labels) — 4-tuple

    Returns:
        spatial_or_inputs: (B, C, H, W) tensor on device
        cond: (B, C_cond) tensor on device, or None if not conditioned
        targets: (B, 1, H, W) tensor on device
        labels: tuple of strings
    """
    if is_conditioned:
        spatial, cond, targets, labels = batch
        spatial = spatial.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        return spatial, cond, targets, labels
    else:
        inputs, targets, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        return inputs, None, targets, labels


def forward_pass(encoder, decoder, spatial_or_inputs, cond=None):
    """
    Forward pass through encoder → decoder, handling both standard and
    conditioned architectures.

    Args:
        encoder: StandardEncoder or ConditionedEncoder
        decoder: StandardDecoder or ConditionedDecoder
        spatial_or_inputs: (B, C, H, W) — spatial features or full input
        cond: (B, C_cond) conditioning vector, or None for standard

    Returns:
        pred: (B, 1, H, W) model prediction
    """
    if cond is not None:
        enc_out, skips = encoder(spatial_or_inputs, cond)
        return decoder(enc_out, skips, cond)
    else:
        enc_out, skips = encoder(spatial_or_inputs)
        return decoder(enc_out, skips)


def augment_conditioned_batch(spatial, cond, targets, slope_dir_channel,
                               augment_fn):
    """
    Apply augmentation to conditioned batch.

    Spatial channels and targets are augmented (flip/rotate).
    Conditioning vector is NOT augmented (scalars, no spatial structure).

    Args:
        spatial: (B, C_spatial, H, W) spatial input features
        cond: (B, C_cond) conditioning vector (returned unchanged)
        targets: (B, 1, H, W) target LST
        slope_dir_channel: index of slope_direction in the SPATIAL channel
            ordering (NOT the original combined ordering). For the v8 dataset
            with spatial=[lc, alb, elev, slpmag, slpdir, urban, suburban, hot],
            this is 4.
        augment_fn: the augment_batch function from unet.py

    Returns:
        (spatial_aug, cond, targets_aug) — cond is unchanged
    """
    spatial_aug, targets_aug = augment_fn(
        spatial, targets, slope_dir_channel=slope_dir_channel)
    return spatial_aug, cond, targets_aug


def split_for_scoring(inputs, cond_channel_indices):
    """
    Split a full input tensor into spatial and conditioning for scoring.

    Used in apply_cae when a conditioned model is loaded and must process
    raw .nc-derived input tensors that contain all channels.

    Args:
        inputs: (B, C_total, H, W) full input tensor with all channels
        cond_channel_indices: list of int, indices of conditioning channels

    Returns:
        spatial: (B, C_spatial, H, W) spatial channels only
        cond: (B, C_cond) conditioning scalars extracted from pixel [0,0]
    """
    n_channels = inputs.shape[1]
    spatial_indices = [i for i in range(n_channels)
                       if i not in cond_channel_indices]

    spatial = inputs[:, spatial_indices, :, :]
    cond = inputs[:, cond_channel_indices, 0, 0]  # broadcast → scalar

    return spatial, cond


# =====================================================================
# Example: how train_from_datasets would look with conditioned support
# =====================================================================
#
# This is a sketch of the key parts of the training loop, showing where
# the helpers are called. Not a complete implementation — just the
# integration points.
#
# def train_from_datasets(self, train_ds, test_ds, model_path, ...):
#     ...
#     is_cond = (self.architecture == 'conditioned')
#
#     # --- Encoder/Decoder creation ---
#     if is_cond:
#         spatial_ch = train_ds.get_input_shape()[0]  # C_spatial
#         cond_dim = train_ds.get_cond_dim()
#         self.encoder, self.decoder = create_conditioned_encoder_decoder(
#             spatial_in_channels=spatial_ch,
#             cond_dim=cond_dim,
#             output_channels=output_chan,
#             base_channels=self.base_channels,
#             dropout_rate=self.dropout_rate,
#             output_activation=self.output_activation)
#     elif self.architecture == 'standard':
#         self.encoder = StandardEncoder(...)
#         self.decoder = StandardDecoder(...)
#     ...
#
#     # --- Training loop ---
#     for epoch in range(nr_epochs):
#         self.encoder.train()
#         self.decoder.train()
#
#         for batch in train_loader:
#             spatial_or_inputs, cond, targets, labels = unpack_batch(
#                 batch, device, is_cond)
#
#             # Augmentation
#             if self.augment:
#                 if is_cond:
#                     spatial_or_inputs, cond, targets = (
#                         augment_conditioned_batch(
#                             spatial_or_inputs, cond, targets,
#                             slope_dir_channel=self.slope_direction_channel,
#                             augment_fn=augment_batch))
#                 else:
#                     spatial_or_inputs, targets = augment_batch(
#                         spatial_or_inputs, targets,
#                         slope_dir_channel=self.slope_direction_channel)
#
#             optimizer.zero_grad()
#             pred = forward_pass(
#                 self.encoder, self.decoder,
#                 spatial_or_inputs, cond)
#             loss = loss_fn(pred, targets)
#             loss.backward()
#             optimizer.step()
#
#     ...
#
#     # --- Test loop (same pattern) ---
#     with torch.no_grad():
#         for batch in test_loader:
#             spatial_or_inputs, cond, targets, labels = unpack_batch(
#                 batch, device, is_cond)
#             pred = forward_pass(
#                 self.encoder, self.decoder,
#                 spatial_or_inputs, cond)
#             ...
