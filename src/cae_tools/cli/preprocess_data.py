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
3. Optionally filters cloud-contaminated boxes (BEFORE normalization)
4. Computes normalization statistics (or uses external ones)
5. Saves normalized tensors to a single .pt file

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

# Z-score filtering (recommended - season-agnostic, catches structured contamination)
# Removes boxes where z-score of (LST - ERA5) delta is below threshold.
# Filtering is applied BEFORE normalization so stats reflect clean data only.
preprocess_data ... --filter-cloud zscore --zscore-threshold -3.0

# Threshold filtering (explicit physical thresholds)
preprocess_data ... --filter-cloud threshold \
    --era5-warm-threshold 288 \
    --delta-threshold-warm -10 \
    --delta-threshold-cold -15
"""

import argparse
import datetime
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

    # Cloud filtering arguments (applied BEFORE normalization)
    parser.add_argument('--filter-cloud', type=str, choices=['none', 'zscore', 'threshold'], default='none',
                        help='Cloud contamination filter to apply before normalization (default: none)')
    parser.add_argument('--zscore-threshold', type=float, default=-3.0,
                        help='Z-score of delta=(LST-ERA5): boxes below this z-score are removed (default: -3.0)')
    parser.add_argument('--era5-warm-threshold', type=float, default=288.0,
                        help='ERA5 temperature separating summer/winter for threshold filter (default: 288K)')
    parser.add_argument('--delta-threshold-warm', type=float, default=-10.0,
                        help='Delta (LST-ERA5) threshold for warm/summer boxes in threshold filter (default: -10K)')
    parser.add_argument('--delta-threshold-cold', type=float, default=-15.0,
                        help='Delta (LST-ERA5) threshold for cold/winter boxes in threshold filter (default: -15K)')

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
    
    # -------------------------------------------------------------------------
    # Cloud contamination filtering
    # Must happen BEFORE normalization so that normalization statistics are
    # computed on clean data only. Filtering after normalization would use
    # min/max values polluted by the very outliers we want to remove.
    # -------------------------------------------------------------------------
    # cloud_filter_meta collects all filtering stats for JSON export
    cloud_filter_meta = {}

    if args.filter_cloud != 'none':
        if 'era5_skt' not in args.input_variables:
            print("Error: --filter-cloud requires 'era5_skt' in --input-variables")
            sys.exit(1)

        era5_idx = args.input_variables.index('era5_skt')

        # Per-box mean in physical units (data still raw/unnormalized here)
        era5_means = inputs[:, era5_idx, :, :].mean(dim=[1, 2])  # (N,) K
        lst_means  = outputs[:, 0, :, :].mean(dim=[1, 2])         # (N,) K
        deltas     = lst_means - era5_means                        # (N,) LST - ERA5
        # NOTE: delta is used for DETECTION only — output target remains raw LST

        if args.filter_cloud == 'zscore':
            # If use_norm_from is provided, load delta stats from train file so
            # test/val are filtered using the same distribution as training.
            # Never recompute stats from test/val data — that would leak information.
            if args.use_norm_from:
                ext = torch.load(args.use_norm_from, map_location='cpu')
                ext_norm = ext['normalisation_parameters']
                if 'delta_mean' not in ext_norm or 'delta_std' not in ext_norm:
                    print("Error: --use-norm-from file has no saved delta_mean/delta_std. "
                          "Was it preprocessed with --filter-cloud zscore?")
                    sys.exit(1)
                delta_mean = ext_norm['delta_mean']
                delta_std  = ext_norm['delta_std']
                print("\nZ-score filtering (delta = LST - ERA5):")
                print(f"  Using TRAINING distribution: mean={delta_mean:.2f}K, std={delta_std:.2f}K")
                source = 'train_file'
            else:
                delta_mean = deltas.mean().item()
                delta_std  = deltas.std().item()
                print("\nZ-score filtering (delta = LST - ERA5):")
                print(f"  Population: mean={delta_mean:.2f}K, std={delta_std:.2f}K")
                source = 'this_dataset'

            z_scores  = (deltas - delta_mean) / delta_std
            keep_mask = z_scores >= args.zscore_threshold
            n_removed = int((~keep_mask).sum())
            cutoff_k  = delta_mean + args.zscore_threshold * delta_std

            print(f"  Threshold: z < {args.zscore_threshold} => delta < {cutoff_k:.2f}K")
            print(f"  Only cold outliers removed (z < threshold, warm extremes preserved)")
            print(f"  Removed: {n_removed} boxes ({100*n_removed/total_boxes:.2f}%)")

            cloud_filter_meta = {
                'method': 'zscore',
                'zscore_threshold': args.zscore_threshold,
                'cutoff_delta_k': round(cutoff_k, 4),
                'delta_mean_k': round(delta_mean, 4),
                'delta_std_k': round(delta_std, 4),
                'delta_stats_source': source,
                'direction': 'cold_only',
                'n_removed': n_removed,
                'n_total_before': int(total_boxes),
                'pct_removed': round(100 * n_removed / total_boxes, 4),
            }

        elif args.filter_cloud == 'threshold':
            warm_mask   = era5_means > args.era5_warm_threshold
            remove_warm = warm_mask  & (deltas < args.delta_threshold_warm)
            remove_cold = (~warm_mask) & (deltas < args.delta_threshold_cold)
            keep_mask   = ~(remove_warm | remove_cold)
            n_removed   = int((~keep_mask).sum())
            print("\nThreshold filtering (delta = LST - ERA5):")
            print(f"  Removed warm (ERA5>{args.era5_warm_threshold}K, "
                  f"delta<{args.delta_threshold_warm}K): {int(remove_warm.sum())} boxes")
            print(f"  Removed cold (ERA5<={args.era5_warm_threshold}K, "
                  f"delta<{args.delta_threshold_cold}K): {int(remove_cold.sum())} boxes")
            print(f"  Total removed: {n_removed} ({100*n_removed/total_boxes:.2f}%)")

            cloud_filter_meta = {
                'method': 'threshold',
                'era5_warm_threshold_k': args.era5_warm_threshold,
                'delta_threshold_warm_k': args.delta_threshold_warm,
                'delta_threshold_cold_k': args.delta_threshold_cold,
                'direction': 'cold_only',
                'n_removed_warm': int(remove_warm.sum()),
                'n_removed_cold': int(remove_cold.sum()),
                'n_removed': n_removed,
                'n_total_before': int(total_boxes),
                'pct_removed': round(100 * n_removed / total_boxes, 4),
            }

        inputs      = inputs[keep_mask]
        outputs     = outputs[keep_mask]
        total_boxes = int(keep_mask.sum())
        print(f"  Remaining: {total_boxes} boxes\n")

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

    # Save zscore delta stats into normalisation_parameters so test/val sets
    # can use training distribution when filtering (never recompute from test data)
    if args.filter_cloud == 'zscore' and cloud_filter_meta and cloud_filter_meta.get('delta_stats_source') == 'this_dataset':
        normalisation_parameters['delta_mean'] = cloud_filter_meta['delta_mean_k']
        normalisation_parameters['delta_std']  = cloud_filter_meta['delta_std_k']
    
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

    # Write comprehensive metadata JSON alongside the .pt file
    meta = {
        'created': datetime.datetime.now().isoformat(),
        'output_file': args.output_file,
        'n_samples': total_boxes,
        'input_variables': args.input_variables,
        'output_variable': args.output_variable,
        'output_activation': args.output_activation,
        'normalized': not args.no_normalize,
        'predict_delta': args.compute_delta,
        'norm_source': args.use_norm_from if args.use_norm_from else 'this_dataset',
        'filter_cloud': args.filter_cloud,
        'cloud_filter': cloud_filter_meta if cloud_filter_meta else None,
        'normalization': {
            'method': 'min_max',
            'min_output': normalisation_parameters['min_output'],
            'max_output': normalisation_parameters['max_output'],
            'inputs': {v: {'min': normalisation_parameters['min_inputs'][v],
                           'max': normalisation_parameters['max_inputs'][v]}
                       for v in args.input_variables},
        },
    }
    meta_path = os.path.splitext(args.output_file)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {meta_path}")
    
    if args.use_norm_from:
        print(f"\nNOTE: Data normalized using parameters from {args.use_norm_from}")
    else:
        print(f"\nNOTE: This file's normalisation_parameters should be used for test/validation data")


if __name__ == '__main__':
    main()
