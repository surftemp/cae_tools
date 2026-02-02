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
Preprocess netCDF training data into efficient .pt format.

This script:
1. Reads netCDF files one at a time (low memory)
2. Extracts only the needed input/output variables
3. Computes normalization statistics
4. Saves normalized tensors to a single .pt file

Usage:
    preprocess_data --input-files /path/to/train/*.nc \
                    --output-file train_preprocessed.pt \
                    --input-variables land_cover albedo elevation ... \
                    --output-variable ST_slices
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import xarray as xr


def count_total_boxes(files, case_dimension='box'):
    """First pass: count total boxes across all files."""
    total = 0
    boxes_per_file = []
    for f in files:
        ds = xr.open_dataset(f)
        n = ds.dims[case_dimension]
        total += n
        boxes_per_file.append(n)
        ds.close()
    return total, boxes_per_file


def broadcast_scalar_to_spatial(values, y_dim, x_dim):
    """Broadcast scalar values (N,) to spatial (N, 1, y, x)."""
    n = values.shape[0]
    expanded = np.broadcast_to(
        values[:, np.newaxis, np.newaxis, np.newaxis],
        (n, 1, y_dim, x_dim)
    )
    return expanded.copy()  # Copy to make it writable


def main():
    parser = argparse.ArgumentParser(description='Preprocess netCDF data to .pt format')
    
    parser.add_argument('--input-files', nargs='+', required=True,
                        help='Path(s) to netCDF files (supports glob patterns)')
    parser.add_argument('--output-file', required=True,
                        help='Output .pt file path')
    parser.add_argument('--input-variables', nargs='+', required=True,
                        help='Names of input variables')
    parser.add_argument('--output-variable', required=True,
                        help='Name of output variable')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip normalization (save raw values)')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    files = []
    for pattern in args.input_files:
        expanded = sorted(glob.glob(pattern))
        files.extend(expanded)
    
    # Filter out non-.nc files
    files = [f for f in files if f.endswith('.nc')]
    
    if not files:
        print("Error: No netCDF files found")
        sys.exit(1)
    
    print(f"Found {len(files)} netCDF files")
    
    # Get dimensions from first file
    ds_sample = xr.open_dataset(files[0])
    case_dimension = ds_sample[args.output_variable].dims[0]
    y_dim = ds_sample.dims.get('y', 100)
    x_dim = ds_sample.dims.get('x', 100)
    n_inputs = len(args.input_variables)
    ds_sample.close()
    
    print(f"Case dimension: {case_dimension}")
    print(f"Spatial dimensions: {y_dim} x {x_dim}")
    print(f"Input channels: {n_inputs}")
    
    # Count total boxes
    print("Counting total boxes...")
    total_boxes, boxes_per_file = count_total_boxes(files, case_dimension)
    print(f"Total boxes: {total_boxes}")
    
    # Pre-allocate tensors
    print(f"Pre-allocating tensors...")
    print(f"  Inputs: {total_boxes} x {n_inputs} x {y_dim} x {x_dim} = {total_boxes * n_inputs * y_dim * x_dim * 4 / 1e9:.1f} GB")
    print(f"  Outputs: {total_boxes} x 1 x {y_dim} x {x_dim} = {total_boxes * 1 * y_dim * x_dim * 4 / 1e9:.1f} GB")
    
    inputs = torch.zeros((total_boxes, n_inputs, y_dim, x_dim), dtype=torch.float32)
    outputs = torch.zeros((total_boxes, 1, y_dim, x_dim), dtype=torch.float32)
    
    # Load data file by file
    print("Loading data...")
    idx = 0
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        n_boxes = ds.dims[case_dimension]
        
        # Extract input variables
        for var_idx, var_name in enumerate(args.input_variables):
            var_data = ds[var_name].values
            
            # Handle scalar variables (broadcast to spatial)
            if var_data.ndim == 1:  # Shape: (n_boxes,)
                var_data = broadcast_scalar_to_spatial(var_data, y_dim, x_dim)
            elif var_data.ndim == 4:  # Shape: (n_boxes, 1, y, x)
                pass  # Already correct shape
            else:
                raise ValueError(f"Unexpected shape for {var_name}: {var_data.shape}")
            
            inputs[idx:idx+n_boxes, var_idx:var_idx+1, :, :] = torch.from_numpy(var_data)
        
        # Extract output variable
        out_data = ds[args.output_variable].values
        if out_data.ndim == 4:
            outputs[idx:idx+n_boxes, :, :, :] = torch.from_numpy(out_data)
        else:
            raise ValueError(f"Unexpected shape for {args.output_variable}: {out_data.shape}")
        
        idx += n_boxes
        ds.close()
        
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  Processed {i+1}/{len(files)} files ({idx}/{total_boxes} boxes)")
    
    # Compute normalization statistics
    print("Computing normalization statistics...")
    normalisation_parameters = {
        'min_inputs': {},
        'max_inputs': {},
        'min_output': None,
        'max_output': None,
    }
    
    for var_idx, var_name in enumerate(args.input_variables):
        var_data = inputs[:, var_idx, :, :]
        normalisation_parameters['min_inputs'][var_name] = float(var_data.min())
        normalisation_parameters['max_inputs'][var_name] = float(var_data.max())
        print(f"  {var_name}: [{var_data.min():.4f}, {var_data.max():.4f}]")
    
    normalisation_parameters['min_output'] = float(outputs.min())
    normalisation_parameters['max_output'] = float(outputs.max())
    print(f"  {args.output_variable}: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # Normalize data (in-place to save memory)
    if not args.no_normalize:
        print("Normalizing data...")
        for var_idx, var_name in enumerate(args.input_variables):
            min_val = normalisation_parameters['min_inputs'][var_name]
            max_val = normalisation_parameters['max_inputs'][var_name]
            range_val = max_val - min_val
            if range_val > 0:
                inputs[:, var_idx, :, :] = (inputs[:, var_idx, :, :] - min_val) / range_val
            else:
                inputs[:, var_idx, :, :] = 0.0
        
        min_out = normalisation_parameters['min_output']
        max_out = normalisation_parameters['max_output']
        range_out = max_out - min_out
        if range_out > 0:
            outputs = (outputs - min_out) / range_out
    
    # Save to .pt file
    print(f"Saving to {args.output_file}...")
    save_dict = {
        'inputs': inputs,
        'outputs': outputs,
        'normalisation_parameters': normalisation_parameters,
        'input_variables': args.input_variables,
        'output_variable': args.output_variable,
        'n_samples': total_boxes,
        'normalized': not args.no_normalize,
    }
    
    torch.save(save_dict, args.output_file)
    
    file_size = os.path.getsize(args.output_file) / 1e9
    print(f"Done! Output file size: {file_size:.2f} GB")
    print(f"Samples: {total_boxes}")


if __name__ == '__main__':
    main()
