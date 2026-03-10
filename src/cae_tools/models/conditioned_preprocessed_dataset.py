"""
Conditioned Preprocessed Dataset.

Wraps a standard PreprocessedDataset .pt file and splits the input tensor
into spatial channels (genuine 100m structure) and conditioning scalars
(broadcast ERA5/temporal values).

For existing .pt files where conditioning variables are broadcast to 100×100,
this extracts a single scalar per conditioning channel (pixel [0,0]).

For new .pt files that store conditioning as a separate vector, this reads
them directly.

Returns 4-tuples: (spatial, cond, target, label)
    spatial: (C_spatial, H, W)  — channels with real spatial structure
    cond:    (C_cond,)           — conditioning scalars
    target:  (1, H, W)          — output LST
    label:   str                 — sample label (empty string if not available)
"""

import torch
import json
import os


class ConditionedPreprocessedDataset(torch.utils.data.Dataset):
    """
    Dataset that splits preprocessed inputs into spatial and conditioning.

    Works in two modes:
        1. Legacy mode: existing .pt file with all channels as (N, C, H, W).
           Conditioning channels are identified by index and collapsed to
           scalars by reading pixel [0,0] (they're broadcast, so all pixels
           are identical).
        2. Native mode: new .pt file with separate 'spatial_inputs' and
           'conditioning' keys. Spatial is (N, C_s, H, W), conditioning
           is (N, C_c) or (N, C_c, 2, 2) for 2×2 ERA5 grids.

    Args:
        pt_path: path to .pt file
        cond_channel_indices: list of channel indices that are conditioning
            (only used in legacy mode). E.g. [3, 4, 5] for era5_skt, sin_doy,
            cos_doy in the standard v8 channel ordering.
        normalisation_parameters: optional override (for test set to use
            train stats)
        flatten_cond_grids: if True and conditioning is stored as 2×2 grids,
            flatten to (N, C_c*4) vector. Default True.
    """

    def __init__(self, pt_path, cond_channel_indices=None,
                 normalisation_parameters=None, flatten_cond_grids=True):
        print(f"Loading conditioned data from {pt_path}...")
        data = torch.load(pt_path)

        self.input_variables = data['input_variables']
        self.output_variable = data['output_variable']
        self.n_samples = data['n_samples']
        self.normalized = data['normalized']
        self.outputs = data['outputs']
        self.flatten_cond_grids = flatten_cond_grids

        # Determine mode: native (separate keys) or legacy (single inputs tensor)
        if 'spatial_inputs' in data and 'conditioning' in data:
            self._init_native(data)
        else:
            self._init_legacy(data, cond_channel_indices)

        # Normalisation parameters
        if normalisation_parameters is not None:
            self.normalisation_parameters = normalisation_parameters
        else:
            self.normalisation_parameters = data['normalisation_parameters']

        self.output_activation = self.normalisation_parameters.get(
            'output_activation', 'sigmoid')

        # Labels (not always present)
        self.labels = data.get('labels', None)

        print(f"Loaded {self.n_samples} samples")
        print(f"Spatial shape:  {self.spatial_inputs.shape}")
        print(f"Cond shape:     {self.conditioning.shape}")
        print(f"Output shape:   {self.outputs.shape}")
        print(f"Spatial vars:   {self.spatial_variables}")
        print(f"Cond vars:      {self.cond_variables}")

    def _init_native(self, data):
        """Initialise from new-format .pt with separate spatial/conditioning."""
        self.spatial_inputs = data['spatial_inputs']
        self.conditioning = data['conditioning']
        self.spatial_variables = data.get('spatial_variables', [])
        self.cond_variables = data.get('cond_variables', [])
        self.mode = 'native'

        # Flatten 2×2 grids if present: (N, C, 2, 2) → (N, C*4)
        if self.conditioning.dim() == 4 and self.flatten_cond_grids:
            N, C, H, W = self.conditioning.shape
            self.conditioning = self.conditioning.reshape(N, C * H * W)
            print(f"Flattened {C} cond grids of {H}×{W} → {C * H * W}d vector")

    def _init_legacy(self, data, cond_channel_indices):
        """Initialise from old-format .pt by splitting channels."""
        inputs = data['inputs']  # (N, C, H, W)
        n_channels = inputs.shape[1]

        if cond_channel_indices is None:
            raise ValueError(
                "cond_channel_indices required for legacy .pt files. "
                "These are the channel indices of conditioning variables "
                "(e.g. [3, 4, 5] for era5_skt, sin_doy, cos_doy).")

        # Validate indices
        for idx in cond_channel_indices:
            if idx < 0 or idx >= n_channels:
                raise ValueError(
                    f"cond_channel_indices contains {idx}, but data has "
                    f"{n_channels} channels (0-{n_channels-1})")

        # Split channels
        spatial_indices = [i for i in range(n_channels)
                          if i not in cond_channel_indices]

        self.spatial_inputs = inputs[:, spatial_indices, :, :]

        # Extract scalars from broadcast channels: take pixel [0,0]
        self.conditioning = inputs[:, cond_channel_indices, 0, 0]  # (N, C_cond)

        # Track variable names
        self.spatial_variables = [self.input_variables[i]
                                  for i in spatial_indices]
        self.cond_variables = [self.input_variables[i]
                               for i in cond_channel_indices]

        self.mode = 'legacy'
        print(f"Legacy mode: split {n_channels} channels → "
              f"{len(spatial_indices)} spatial + "
              f"{len(cond_channel_indices)} conditioning")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        spatial = self.spatial_inputs[idx]      # (C_spatial, H, W)
        cond = self.conditioning[idx]            # (C_cond,)
        target = self.outputs[idx]               # (1, H, W)
        label = self.labels[idx] if self.labels is not None else ""
        return spatial, cond, target, label

    # ---- Compatibility interface (matches PreprocessedDataset) ----

    def get_normalisation_parameters(self):
        return self.normalisation_parameters

    def set_normalisation_parameters(self, parameters):
        self.normalisation_parameters = parameters

    def set_normalise_output(self, normalise_out):
        self.normalise_out = normalise_out

    def get_input_shape(self):
        """Return spatial input shape: (C_spatial, H, W)."""
        return tuple(self.spatial_inputs.shape[1:])

    def get_cond_dim(self):
        """Return conditioning vector dimension."""
        return self.conditioning.shape[1]

    def get_output_shape(self):
        return tuple(self.outputs.shape[1:])

    def get_input_spec(self):
        """Return input spec for spatial channels only."""
        H, W = self.spatial_inputs.shape[2], self.spatial_inputs.shape[3]
        return [{"name": var, "shape": [1, H, W]}
                for var in self.spatial_variables]

    def get_output_spec(self):
        return {
            "name": self.output_variable,
            "shape": list(self.outputs.shape[1:])
        }

    def get_spatial_variables(self):
        return self.spatial_variables

    def get_cond_variables(self):
        return self.cond_variables
