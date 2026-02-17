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
3. Computes normalization statistics (or uses external ones)
4. Saves normalized tensors to a single .pt file

Usage (training data - computes its own stats):
    preprocess_data --input-files /path/to/train/*.nc \
                    --output-file train_preprocessed.pt \
                    --input-variables land_cover albedo elevation ... \
                    --output-variable ST_slices

Usage (test data - uses train stats):
    preprocess_data --input-files /path/to/test/*.nc \
                    --output-file test_preprocessed.pt \
                    --input-variables land_cover albedo elevation ... \
                    --output-variable ST_slices \
                    --use-norm-from train_preprocessed.pt
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
    parser.add_argument('--use-norm-from', type=str, default=None,
                        help='Path to .pt file to load normalisation parameters from (use for test/val data)')
    parser.add_argument('--compute-delta', action='store_true',
                        help='Compute delta targets: output = LST - ERA5 (for residual learning)')
    parser.add_argument('--delta-reference-variable', type=str, default='era5_skt',
                        help='Input variable to subtract from output when --compute-delta (default: era5_skt)')
    parser.add_argument('--output-activation', type=str, choices=['sigmoid', 'tanh'], default='sigmoid',
                        help='Output activation: sigmoid normalizes to [0,1], tanh to [-1,1] (default: sigmoid)')
    
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
    
    # Compute delta targets if requested (LST - ERA5, per pixel)
    if args.compute_delta:
        ref_var = args.delta_reference_variable
        if ref_var not in args.input_variables:
            print(f"Error: delta reference variable '{ref_var}' not in input variables")
            sys.exit(1)
        ref_idx = args.input_variables.index(ref_var)
        era5_channel = inputs[:, ref_idx:ref_idx+1, :, :]  # (N, 1, y, x)
        print(f"Computing delta targets: {args.output_variable} - {ref_var}")
        print(f"  Output range before delta: [{outputs.min():.2f}, {outputs.max():.2f}]")
        print(f"  ERA5 range: [{era5_channel.min():.2f}, {era5_channel.max():.2f}]")
        outputs = outputs - era5_channel
        print(f"  Delta range: [{outputs.min():.2f}, {outputs.max():.2f}]")

    # Load or compute normalization parameters
    if args.use_norm_from:
        # Load normalisation parameters from external .pt file (e.g., training data)
        print(f"Loading normalisation parameters from {args.use_norm_from}...")
        external_data = torch.load(args.use_norm_from, map_location='cpu')
        normalisation_parameters = external_data['normalisation_parameters']
        
        print("Using external normalisation parameters:")
        for var_name in args.input_variables:
            min_val = normalisation_parameters['min_inputs'][var_name]
            max_val = normalisation_parameters['max_inputs'][var_name]
            print(f"  {var_name}: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  {args.output_variable}: [{normalisation_parameters['min_output']:.4f}, {normalisation_parameters['max_output']:.4f}]")
        
        # Also report this dataset's actual min/max for comparison
        print("This dataset's actual ranges (for reference):")
        for var_idx, var_name in enumerate(args.input_variables):
            var_data = inputs[:, var_idx, :, :]
            print(f"  {var_name}: [{var_data.min():.4f}, {var_data.max():.4f}]")
        print(f"  {args.output_variable}: [{outputs.min():.4f}, {outputs.max():.4f}]")
    else:
        # Compute normalization statistics from this dataset
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
    
    # Store output activation and delta metadata in normalisation_parameters
    normalisation_parameters['output_activation'] = args.output_activation
    if args.compute_delta:
        normalisation_parameters['predict_delta'] = True
        normalisation_parameters['delta_reference_variable'] = args.delta_reference_variable
    
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
            if args.output_activation == 'tanh':
                outputs = 2 * (outputs - min_out) / range_out - 1  # → [-1, 1]
                print(f"  Output normalized to [-1, 1] (tanh mode)")
            else:
                outputs = (outputs - min_out) / range_out  # → [0, 1]
                print(f"  Output normalized to [0, 1] (sigmoid mode)")
    
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
        'output_activation': args.output_activation,
        'predict_delta': args.compute_delta,
        'delta_reference_variable': args.delta_reference_variable if args.compute_delta else None,
    }
    
    torch.save(save_dict, args.output_file)
    
    file_size = os.path.getsize(args.output_file) / 1e9
    print(f"Done! Output file size: {file_size:.2f} GB")
    print(f"Samples: {total_boxes}")
    
    if args.use_norm_from:
        print(f"\nNOTE: Data normalized using parameters from {args.use_norm_from}")
    else:
        print(f"\nNOTE: This file's normalisation_parameters should be used for test/validation data")


if __name__ == '__main__':
    main()
