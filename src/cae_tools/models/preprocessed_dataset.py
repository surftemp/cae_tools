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

"""
Simple PyTorch Dataset for preprocessed .pt files.

This is much faster than DSDataset because:
- No xarray overhead
- No per-sample normalization (already normalized)
- Direct tensor slicing
"""

import numpy as np
import torch


class PreprocessedDataset(torch.utils.data.Dataset):
    """Dataset for loading preprocessed .pt files."""
    
    def __init__(self, pt_path, normalisation_parameters=None):
        """
        Args:
            pt_path: Path to .pt file created by preprocess_data.py
            normalisation_parameters: Optional override (for test set to use train stats)
        """
        print(f"Loading preprocessed data from {pt_path}...")
        data = torch.load(pt_path)
        
        self.inputs = data['inputs']
        self.outputs = data['outputs']
        self.input_variables = data['input_variables']
        self.output_variable = data['output_variable']
        self.n_samples = data['n_samples']
        self.normalized = data['normalized']
        
        # For compatibility with DSDataset
        self.transform = None
        self.normalise_out = True
        
        # Use provided normalisation_parameters or load from file
        if normalisation_parameters is not None:
            self.normalisation_parameters = normalisation_parameters
        else:
            self.normalisation_parameters = data['normalisation_parameters']
        
        self.output_activation = self.normalisation_parameters.get('output_activation', 'sigmoid')
        
        print(f"Loaded {self.n_samples} samples")
        print(f"Input shape: {self.inputs.shape}")
        print(f"Output shape: {self.outputs.shape}")
    
    def get_normalisation_parameters(self):
        return self.normalisation_parameters
    
    def set_normalisation_parameters(self, parameters):
        """For compatibility with DSDataset interface."""
        self.normalisation_parameters = parameters
    
    def set_normalise_output(self, normalise_out):
        """For compatibility with DSDataset interface (used by evaluate())."""
        self.normalise_out = normalise_out
    
    def get_input_shape(self):
        return tuple(self.inputs.shape[1:])  # (channels, y, x)
    
    def get_output_shape(self):
        return tuple(self.outputs.shape[1:])  # (channels, y, x)
    
    def get_input_spec(self):
        """Return input spec in same format as DSDataset."""
        input_spec = []
        for var_name in self.input_variables:
            input_spec.append({
                "name": var_name,
                "shape": [1, self.inputs.shape[2], self.inputs.shape[3]]
            })
        return input_spec
    
    def get_output_spec(self):
        """Return output spec in same format as DSDataset."""
        return {
            "name": self.output_variable,
            "shape": list(self.outputs.shape[1:])
        }
    
    def denormalise_output(self, arr, force=False):
        """Denormalize output back to original scale."""
        min_out = self.normalisation_parameters['min_output']
        max_out = self.normalisation_parameters['max_output']
        range_out = max_out - min_out
        output_activation = self.normalisation_parameters.get('output_activation', 'sigmoid')
        if output_activation == 'tanh':
            return min_out + ((arr + 1) / 2) * range_out  # [-1,1] → physical
        else:
            return min_out + (arr * range_out)  # [0,1] → physical
    
    def __getitem__(self, index):
        label = f"image{index}"
        
        # Return normalized or raw output based on normalise_out flag
        if self.normalise_out:
            out = self.outputs[index]
        else:
            # Denormalize on-the-fly for evaluate()
            out_np = self.denormalise_output(self.outputs[index].numpy(), force=True)
            out = torch.tensor(out_np, dtype=torch.float32)
        
        return (self.inputs[index], out, label)
    
    def __len__(self):
        return self.n_samples
