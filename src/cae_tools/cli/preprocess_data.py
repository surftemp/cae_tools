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

Supports two output formats:

LEGACY FORMAT (--input-variables):
    All variables broadcast to (N, C, 100, 100) and stored under 'inputs'.
    Used by PreprocessedDataset.

CONDITIONED FORMAT (--spatial-variables + --cond-variables):
    Spatial variables stored as (N, C_spatial, 100, 100) under 'spatial_inputs'.
    Conditioning variables stored as (N, C_cond) scalars under 'conditioning'.
    Conditioning variables are NOT broadcast to 100x100, saving ~40% disk space.
    Used by ConditionedPreprocessedDataset.

Usage (legacy - all broadcast):
    preprocess_data --input-files /path/to/train/*.nc \\
                    --output-file train_preprocessed.pt \\
                    --input-variables land_cover albedo elevation era5_skt ... \\
                    --output-variable ST_slices

Usage (conditioned - separate spatial and conditioning):
    preprocess_data --input-files /path/to/train/*.nc \\
                    --output-file train_v9_conditioned.pt \\
                    --spatial-variables land_cover albedo_monthly_climatology_means \\
                        elevation slope_magnitude slope_direction \\
                        urban_area suburban_area pixel_st_hot_pattern \\
                    --cond-variables era5_skt era5_d2m era5_u10 era5_v10 \\
                        era5_ssrd era5_tp era5_slhf era5_stl1 era5_lai_hv \\
                        sin_doy cos_doy \\
                    --output-variable ST_slices

Usage (test/val data - uses train stats for both formats):
    preprocess_data --input-files /path/to/test/*.nc \\
                    --output-file test_v9_conditioned.pt \\
                    --spatial-variables ... --cond-variables ... \\
                    --output-variable ST_slices \\
                    --use-norm-from train_v9_conditioned.pt

# Z-score filtering (recommended - season-agnostic, catches structured contamination)
# Removes boxes where z-score of (LST - ERA5) delta is below threshold.
# Filtering is applied BEFORE normalization so stats reflect clean data only.
preprocess_data ... --filter-cloud zscore --zscore-threshold -3.0

# Threshold filtering (explicit physical thresholds)
preprocess_data ... --filter-cloud threshold \\
    --era5-warm-threshold 288 \\
    --delta-threshold-warm -10 \\
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
    parser.add_argument('--output-variable', required=True,
                        help='Name of output variable')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip normalization (save raw values)')
    parser.add_argument('--use-norm-from', type=str, default=None,
                        help='Path to .pt file to load normalisation parameters from '
                             '(use for test/val data)')
    parser.add_argument('--compute-delta', action='store_true',
                        help='Compute delta targets: output = LST - ERA5 '
                             '(for residual learning)')
    parser.add_argument('--delta-reference-variable', type=str, default='era5_skt',
                        help='Input variable to subtract from output when '
                             '--compute-delta (default: era5_skt)')
    parser.add_argument('--output-activation', type=str,
                        choices=['sigmoid', 'tanh', 'none'], default='sigmoid',
                        help='Output activation: sigmoid normalizes to [0,1], '
                             'tanh to [-1,1] (default: sigmoid)')

    # --- Variable specification (two mutually exclusive modes) ---
    # Legacy mode: all variables in one list, all broadcast to spatial
    parser.add_argument('--input-variables', nargs='+', required=False,
                        help='(Legacy mode) Names of ALL input variables. '
                             'Every variable is broadcast to (N,1,H,W).')

    # Conditioned mode: separate spatial and conditioning variables
    parser.add_argument('--spatial-variables', nargs='+', required=False,
                        help='(Conditioned mode) Variables with real 100m spatial '
                             'structure (e.g. land_cover, elevation, slope). '
                             'Stored as (N, C_spatial, H, W).')
    parser.add_argument('--cond-variables', nargs='+', required=False,
                        help='(Conditioned mode) Conditioning variables that are '
                             'scalars per box (e.g. ERA5 fields, sin_doy, cos_doy). '
                             'Stored as (N, C_cond) without spatial broadcast. '
                             'Saves ~40%% disk space vs broadcasting.')

    # Cloud filtering arguments (applied BEFORE normalization)
    parser.add_argument('--filter-cloud', type=str,
                        choices=['none', 'zscore', 'threshold'], default='none',
                        help='Cloud contamination filter to apply before normalization')
    parser.add_argument('--zscore-threshold', type=float, default=-3.0,
                        help='Z-score threshold for cloud filtering (default: -3.0)')
    parser.add_argument('--era5-warm-threshold', type=float, default=288.0,
                        help='ERA5 temperature threshold separating warm/cold '
                             'seasons for threshold filtering (default: 288K)')
    parser.add_argument('--delta-threshold-warm', type=float, default=-10.0,
                        help='Delta (LST-ERA5) threshold for warm season '
                             '(default: -10K)')
    parser.add_argument('--delta-threshold-cold', type=float, default=-15.0,
                        help='Delta (LST-ERA5) threshold for cold season '
                             '(default: -15K)')

    args = parser.parse_args()

    # =====================================================================
    # Determine mode: legacy or conditioned
    # =====================================================================
    has_legacy = args.input_variables is not None
    has_conditioned = (args.spatial_variables is not None or
                       args.cond_variables is not None)

    if has_legacy and has_conditioned:
        print("Error: cannot use --input-variables together with "
              "--spatial-variables / --cond-variables. Choose one mode.")
        sys.exit(1)

    if not has_legacy and not has_conditioned:
        print("Error: must specify either --input-variables (legacy mode) or "
              "--spatial-variables + --cond-variables (conditioned mode).")
        sys.exit(1)

    if has_conditioned:
        if args.spatial_variables is None or args.cond_variables is None:
            print("Error: conditioned mode requires both --spatial-variables "
                  "and --cond-variables.")
            sys.exit(1)
        conditioned_mode = True
        # For internal processing, combine into a single ordered list
        # so cloud filtering / delta computation can find variables by name.
        all_input_variables = args.spatial_variables + args.cond_variables
        n_spatial = len(args.spatial_variables)
        n_cond = len(args.cond_variables)
        print(f"Conditioned mode: {n_spatial} spatial + {n_cond} conditioning "
              f"= {len(all_input_variables)} total input variables")
        print(f"  Spatial:      {args.spatial_variables}")
        print(f"  Conditioning: {args.cond_variables}")
    else:
        conditioned_mode = False
        all_input_variables = args.input_variables
        n_spatial = len(all_input_variables)
        n_cond = 0
        print(f"Legacy mode: {n_spatial} input variables (all broadcast)")

    # =====================================================================
    # Expand glob patterns and count boxes
    # =====================================================================
    files = []
    for pattern in args.input_files:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"Warning: no files match pattern '{pattern}'")
        files.extend(expanded)

    if not files:
        print("Error: no input files found")
        sys.exit(1)

    print(f"\nFound {len(files)} input files")

    # Determine case dimension (some datasets use 'box', others may differ)
    sample_ds = xr.open_dataset(files[0])
    if 'box' in sample_ds.dims:
        case_dimension = 'box'
    elif 'case' in sample_ds.dims:
        case_dimension = 'case'
    else:
        case_dimension = list(sample_ds.dims.keys())[0]
        print(f"Warning: using first dimension '{case_dimension}' as case dimension")

    # Get spatial dimensions from first spatial variable
    first_spatial_var = (args.spatial_variables[0] if conditioned_mode
                         else all_input_variables[0])
    first_var_data = sample_ds[first_spatial_var]
    if first_var_data.ndim == 4:  # (box, channel, y, x)
        y_dim = first_var_data.shape[2]
        x_dim = first_var_data.shape[3]
    elif first_var_data.ndim == 1:  # scalar — look for a spatial variable
        for vname in (args.spatial_variables if conditioned_mode
                      else all_input_variables):
            vdata = sample_ds[vname]
            if vdata.ndim == 4:
                y_dim = vdata.shape[2]
                x_dim = vdata.shape[3]
                break
        else:
            print("Error: could not determine spatial dimensions — "
                  "no 4D variable found.")
            sys.exit(1)
    else:
        y_dim = first_var_data.shape[-2]
        x_dim = first_var_data.shape[-1]
    sample_ds.close()

    print(f"Spatial dimensions: {y_dim} x {x_dim}")
    print(f"Case dimension: '{case_dimension}'")

    total_boxes, boxes_per_file = count_total_boxes(files, case_dimension)
    n_inputs = len(all_input_variables)
    print(f"Total boxes: {total_boxes}")

    # =====================================================================
    # Allocate storage
    # =====================================================================
    if conditioned_mode:
        # Spatial: (N, C_spatial, H, W)
        # Conditioning: (N, C_cond) — scalar per box per variable
        spatial_size_gb = total_boxes * n_spatial * y_dim * x_dim * 4 / 1e9
        cond_size_gb = total_boxes * n_cond * 4 / 1e9
        out_size_gb = total_boxes * 1 * y_dim * x_dim * 4 / 1e9
        print(f"\nAllocating tensors:")
        print(f"  Spatial:      {total_boxes} x {n_spatial} x {y_dim} x "
              f"{x_dim} = {spatial_size_gb:.1f} GB")
        print(f"  Conditioning: {total_boxes} x {n_cond} = {cond_size_gb:.4f} GB")
        print(f"  Outputs:      {total_boxes} x 1 x {y_dim} x {x_dim} = "
              f"{out_size_gb:.1f} GB")

        spatial_inputs = torch.zeros(
            (total_boxes, n_spatial, y_dim, x_dim), dtype=torch.float32)
        conditioning = torch.zeros(
            (total_boxes, n_cond), dtype=torch.float32)
    else:
        total_size_gb = total_boxes * n_inputs * y_dim * x_dim * 4 / 1e9
        out_size_gb = total_boxes * 1 * y_dim * x_dim * 4 / 1e9
        print(f"\nAllocating tensors:")
        print(f"  Inputs:  {total_boxes} x {n_inputs} x {y_dim} x {x_dim} "
              f"= {total_size_gb:.1f} GB")
        print(f"  Outputs: {total_boxes} x 1 x {y_dim} x {x_dim} = "
              f"{out_size_gb:.1f} GB")

        inputs = torch.zeros(
            (total_boxes, n_inputs, y_dim, x_dim), dtype=torch.float32)

    outputs = torch.zeros(
        (total_boxes, 1, y_dim, x_dim), dtype=torch.float32)

    # =====================================================================
    # Load data file by file
    # =====================================================================
    print("Loading data...")
    idx = 0
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        n_boxes = ds.dims[case_dimension]

        if conditioned_mode:
            # Load spatial variables → spatial_inputs tensor
            for var_idx, var_name in enumerate(args.spatial_variables):
                var_data = ds[var_name].values
                if var_data.ndim == 1:  # (n_boxes,) — scalar, broadcast
                    var_data = broadcast_scalar_to_spatial(
                        var_data, y_dim, x_dim)
                elif var_data.ndim != 4:
                    raise ValueError(
                        f"Unexpected shape for spatial var {var_name}: "
                        f"{var_data.shape}")
                spatial_inputs[idx:idx+n_boxes, var_idx:var_idx+1, :, :] = \
                    torch.from_numpy(var_data)

            # Load conditioning variables → conditioning tensor (scalars)
            for var_idx, var_name in enumerate(args.cond_variables):
                var_data = ds[var_name].values
                if var_data.ndim == 1:  # (n_boxes,) — already scalar
                    conditioning[idx:idx+n_boxes, var_idx] = \
                        torch.from_numpy(var_data)
                elif var_data.ndim == 4:  # (n_boxes, 1, y, x) — extract mean
                    # ERA5 variables are broadcast to spatial in .nc files.
                    # All pixels have the same value, so take [0,0].
                    conditioning[idx:idx+n_boxes, var_idx] = \
                        torch.from_numpy(var_data[:, 0, 0, 0])
                else:
                    raise ValueError(
                        f"Unexpected shape for cond var {var_name}: "
                        f"{var_data.shape}")
        else:
            # Legacy mode: all variables broadcast to spatial
            for var_idx, var_name in enumerate(all_input_variables):
                var_data = ds[var_name].values
                if var_data.ndim == 1:
                    var_data = broadcast_scalar_to_spatial(
                        var_data, y_dim, x_dim)
                elif var_data.ndim != 4:
                    raise ValueError(
                        f"Unexpected shape for {var_name}: {var_data.shape}")
                inputs[idx:idx+n_boxes, var_idx:var_idx+1, :, :] = \
                    torch.from_numpy(var_data)

        # Extract output variable
        out_data = ds[args.output_variable].values
        if out_data.ndim == 4:
            outputs[idx:idx+n_boxes, :, :, :] = torch.from_numpy(out_data)
        else:
            raise ValueError(
                f"Unexpected shape for {args.output_variable}: "
                f"{out_data.shape}")

        idx += n_boxes
        ds.close()

        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  Processed {i+1}/{len(files)} files "
                  f"({idx}/{total_boxes} boxes)")

    # =====================================================================
    # Cloud contamination filtering
    # Must happen BEFORE normalization so that normalization statistics are
    # computed on clean data only.
    # =====================================================================
    cloud_filter_meta = {}

    if args.filter_cloud != 'none':
        # Find era5_skt wherever it is
        if 'era5_skt' not in all_input_variables:
            print("Error: --filter-cloud requires 'era5_skt' in input variables")
            sys.exit(1)

        # Get ERA5 values as (N,) in physical units
        if conditioned_mode and 'era5_skt' in args.cond_variables:
            era5_cond_idx = args.cond_variables.index('era5_skt')
            era5_means = conditioning[:, era5_cond_idx]  # already scalar
        elif conditioned_mode and 'era5_skt' in args.spatial_variables:
            era5_sp_idx = args.spatial_variables.index('era5_skt')
            era5_means = spatial_inputs[:, era5_sp_idx, :, :].mean(dim=[1, 2])
        else:
            # Legacy mode
            era5_idx = all_input_variables.index('era5_skt')
            era5_means = inputs[:, era5_idx, :, :].mean(dim=[1, 2])

        lst_means = outputs[:, 0, :, :].mean(dim=[1, 2])
        deltas = lst_means - era5_means

        if args.filter_cloud == 'zscore':
            if args.use_norm_from:
                ext = torch.load(args.use_norm_from, map_location='cpu')
                ext_norm = ext['normalisation_parameters']
                if 'delta_mean' not in ext_norm or 'delta_std' not in ext_norm:
                    print("Error: --use-norm-from file has no saved "
                          "delta_mean/delta_std. Was it preprocessed with "
                          "--filter-cloud zscore?")
                    sys.exit(1)
                delta_mean = ext_norm['delta_mean']
                delta_std = ext_norm['delta_std']
                print("\nZ-score filtering (delta = LST - ERA5):")
                print(f"  Using TRAINING distribution: "
                      f"mean={delta_mean:.2f}K, std={delta_std:.2f}K")
                source = 'train_file'
            else:
                delta_mean = deltas.mean().item()
                delta_std = deltas.std().item()
                print("\nZ-score filtering (delta = LST - ERA5):")
                print(f"  Population: mean={delta_mean:.2f}K, "
                      f"std={delta_std:.2f}K")
                source = 'this_dataset'

            z_scores = (deltas - delta_mean) / delta_std
            keep_mask = z_scores >= args.zscore_threshold
            n_removed = int((~keep_mask).sum())
            cutoff_k = delta_mean + args.zscore_threshold * delta_std

            print(f"  Threshold: z < {args.zscore_threshold} "
                  f"=> delta < {cutoff_k:.2f}K")
            print(f"  Removed: {n_removed} boxes "
                  f"({100*n_removed/total_boxes:.2f}%)")

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
            warm_mask = era5_means > args.era5_warm_threshold
            remove_warm = warm_mask & (deltas < args.delta_threshold_warm)
            remove_cold = (~warm_mask) & (deltas < args.delta_threshold_cold)
            keep_mask = ~(remove_warm | remove_cold)
            n_removed = int((~keep_mask).sum())
            print("\nThreshold filtering (delta = LST - ERA5):")
            print(f"  Removed warm (ERA5>{args.era5_warm_threshold}K, "
                  f"delta<{args.delta_threshold_warm}K): "
                  f"{int(remove_warm.sum())} boxes")
            print(f"  Removed cold (ERA5<={args.era5_warm_threshold}K, "
                  f"delta<{args.delta_threshold_cold}K): "
                  f"{int(remove_cold.sum())} boxes")
            print(f"  Total removed: {n_removed} "
                  f"({100*n_removed/total_boxes:.2f}%)")

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

        # Apply filter
        if conditioned_mode:
            spatial_inputs = spatial_inputs[keep_mask]
            conditioning = conditioning[keep_mask]
        else:
            inputs = inputs[keep_mask]
        outputs = outputs[keep_mask]
        total_boxes = int(keep_mask.sum())
        print(f"  Remaining: {total_boxes} boxes\n")

    # =====================================================================
    # Compute delta targets if requested (LST - ERA5, per pixel)
    # =====================================================================
    if args.compute_delta:
        ref_var = args.delta_reference_variable
        if ref_var not in all_input_variables:
            print(f"Error: delta reference variable '{ref_var}' not in "
                  "input variables")
            sys.exit(1)

        # Get ERA5 as (N, 1, H, W) for per-pixel subtraction
        if conditioned_mode and ref_var in args.cond_variables:
            cond_idx = args.cond_variables.index(ref_var)
            era5_channel = conditioning[:, cond_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            era5_channel = era5_channel.expand(-1, 1, y_dim, x_dim)
        elif conditioned_mode and ref_var in args.spatial_variables:
            sp_idx = args.spatial_variables.index(ref_var)
            era5_channel = spatial_inputs[:, sp_idx:sp_idx+1, :, :]
        else:
            ref_idx = all_input_variables.index(ref_var)
            era5_channel = inputs[:, ref_idx:ref_idx+1, :, :]

        print(f"Computing delta targets: {args.output_variable} - {ref_var}")
        print(f"  Output range before delta: "
              f"[{outputs.min():.2f}, {outputs.max():.2f}]")
        print(f"  ERA5 range: "
              f"[{era5_channel.min():.2f}, {era5_channel.max():.2f}]")
        outputs = outputs - era5_channel
        print(f"  Delta range: "
              f"[{outputs.min():.2f}, {outputs.max():.2f}]")

    # =====================================================================
    # Load or compute normalization parameters
    # =====================================================================
    if args.use_norm_from:
        print(f"Loading normalisation parameters from {args.use_norm_from}...")
        external_data = torch.load(args.use_norm_from, map_location='cpu')
        normalisation_parameters = external_data['normalisation_parameters']

        print("Using external normalisation parameters:")
        for var_name in all_input_variables:
            min_val = normalisation_parameters['min_inputs'][var_name]
            max_val = normalisation_parameters['max_inputs'][var_name]
            print(f"  {var_name}: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  {args.output_variable}: "
              f"[{normalisation_parameters['min_output']:.4f}, "
              f"{normalisation_parameters['max_output']:.4f}]")

        # Report this dataset's actual ranges for comparison
        print("This dataset's actual ranges (for reference):")
        if conditioned_mode:
            for var_idx, var_name in enumerate(args.spatial_variables):
                vd = spatial_inputs[:, var_idx, :, :]
                print(f"  {var_name}: [{vd.min():.4f}, {vd.max():.4f}]")
            for var_idx, var_name in enumerate(args.cond_variables):
                vd = conditioning[:, var_idx]
                print(f"  {var_name}: [{vd.min():.4f}, {vd.max():.4f}]")
        else:
            for var_idx, var_name in enumerate(all_input_variables):
                vd = inputs[:, var_idx, :, :]
                print(f"  {var_name}: [{vd.min():.4f}, {vd.max():.4f}]")
        print(f"  {args.output_variable}: "
              f"[{outputs.min():.4f}, {outputs.max():.4f}]")
    else:
        # Compute normalization statistics from this dataset
        print("Computing normalization statistics...")
        normalisation_parameters = {
            'min_inputs': {},
            'max_inputs': {},
            'min_output': None,
            'max_output': None,
        }

        if conditioned_mode:
            for var_idx, var_name in enumerate(args.spatial_variables):
                vd = spatial_inputs[:, var_idx, :, :]
                normalisation_parameters['min_inputs'][var_name] = \
                    float(vd.min())
                normalisation_parameters['max_inputs'][var_name] = \
                    float(vd.max())
                print(f"  {var_name}: [{vd.min():.4f}, {vd.max():.4f}]")
            for var_idx, var_name in enumerate(args.cond_variables):
                vd = conditioning[:, var_idx]
                normalisation_parameters['min_inputs'][var_name] = \
                    float(vd.min())
                normalisation_parameters['max_inputs'][var_name] = \
                    float(vd.max())
                print(f"  {var_name}: [{vd.min():.4f}, {vd.max():.4f}]")
        else:
            for var_idx, var_name in enumerate(all_input_variables):
                vd = inputs[:, var_idx, :, :]
                normalisation_parameters['min_inputs'][var_name] = \
                    float(vd.min())
                normalisation_parameters['max_inputs'][var_name] = \
                    float(vd.max())
                print(f"  {var_name}: [{vd.min():.4f}, {vd.max():.4f}]")

        normalisation_parameters['min_output'] = float(outputs.min())
        normalisation_parameters['max_output'] = float(outputs.max())
        print(f"  {args.output_variable}: "
              f"[{outputs.min():.4f}, {outputs.max():.4f}]")

    # Store output activation and delta metadata
    normalisation_parameters['output_activation'] = args.output_activation
    if args.compute_delta:
        normalisation_parameters['predict_delta'] = True
        normalisation_parameters['delta_reference_variable'] = \
            args.delta_reference_variable

    # Save zscore delta stats so test/val can use training distribution
    if (args.filter_cloud == 'zscore' and cloud_filter_meta and
            cloud_filter_meta.get('delta_stats_source') == 'this_dataset'):
        normalisation_parameters['delta_mean'] = \
            cloud_filter_meta['delta_mean_k']
        normalisation_parameters['delta_std'] = \
            cloud_filter_meta['delta_std_k']

    # =====================================================================
    # Normalize data (in-place to save memory)
    # =====================================================================
    if not args.no_normalize:
        print("Normalizing data...")

        if conditioned_mode:
            # Normalize spatial variables
            for var_idx, var_name in enumerate(args.spatial_variables):
                min_val = normalisation_parameters['min_inputs'][var_name]
                max_val = normalisation_parameters['max_inputs'][var_name]
                range_val = max_val - min_val
                if range_val > 0:
                    spatial_inputs[:, var_idx, :, :] = \
                        (spatial_inputs[:, var_idx, :, :] - min_val) / range_val
                else:
                    spatial_inputs[:, var_idx, :, :] = 0.0

            # Normalize conditioning variables
            for var_idx, var_name in enumerate(args.cond_variables):
                min_val = normalisation_parameters['min_inputs'][var_name]
                max_val = normalisation_parameters['max_inputs'][var_name]
                range_val = max_val - min_val
                if range_val > 0:
                    conditioning[:, var_idx] = \
                        (conditioning[:, var_idx] - min_val) / range_val
                else:
                    conditioning[:, var_idx] = 0.0
        else:
            for var_idx, var_name in enumerate(all_input_variables):
                min_val = normalisation_parameters['min_inputs'][var_name]
                max_val = normalisation_parameters['max_inputs'][var_name]
                range_val = max_val - min_val
                if range_val > 0:
                    inputs[:, var_idx, :, :] = \
                        (inputs[:, var_idx, :, :] - min_val) / range_val
                else:
                    inputs[:, var_idx, :, :] = 0.0

        # Normalize output
        min_out = normalisation_parameters['min_output']
        max_out = normalisation_parameters['max_output']
        range_out = max_out - min_out
        if range_out > 0:
            if args.output_activation == 'tanh':
                outputs = 2 * (outputs - min_out) / range_out - 1
                print(f"  Output normalized to [-1, 1] (tanh mode)")
            else:
                outputs = (outputs - min_out) / range_out
                print(f"  Output normalized to [0, 1] (sigmoid mode)")

    # =====================================================================
    # Save to .pt file
    # =====================================================================
    print(f"Saving to {args.output_file}...")

    if conditioned_mode:
        save_dict = {
            # Conditioned format keys
            'spatial_inputs': spatial_inputs,
            'conditioning': conditioning,
            'spatial_variables': args.spatial_variables,
            'cond_variables': args.cond_variables,
            # Also store combined list as 'input_variables' for compatibility
            # with code that reads this key (e.g. normalisation lookups)
            'input_variables': all_input_variables,
            'outputs': outputs,
            'normalisation_parameters': normalisation_parameters,
            'output_variable': args.output_variable,
            'n_samples': total_boxes,
            'normalized': not args.no_normalize,
            'output_activation': args.output_activation,
            'format': 'conditioned',
            'predict_delta': args.compute_delta,
            'delta_reference_variable': (args.delta_reference_variable
                                         if args.compute_delta else None),
        }
    else:
        save_dict = {
            'inputs': inputs,
            'outputs': outputs,
            'normalisation_parameters': normalisation_parameters,
            'input_variables': all_input_variables,
            'output_variable': args.output_variable,
            'n_samples': total_boxes,
            'normalized': not args.no_normalize,
            'output_activation': args.output_activation,
            'format': 'legacy',
            'predict_delta': args.compute_delta,
            'delta_reference_variable': (args.delta_reference_variable
                                         if args.compute_delta else None),
        }

    if cloud_filter_meta:
        save_dict['cloud_filter_meta'] = cloud_filter_meta

    torch.save(save_dict, args.output_file)

    file_size = os.path.getsize(args.output_file) / 1e9
    print(f"Done! Saved {total_boxes} samples to {args.output_file} "
          f"({file_size:.2f} GB)")

    # Also save metadata as JSON for quick inspection without loading tensor
    meta_path = args.output_file.replace('.pt', '_meta.json')
    meta = {
        'format': 'conditioned' if conditioned_mode else 'legacy',
        'n_samples': total_boxes,
        'normalized': not args.no_normalize,
        'output_activation': args.output_activation,
        'output_variable': args.output_variable,
        'normalisation_parameters': normalisation_parameters,
    }
    if conditioned_mode:
        meta['spatial_variables'] = args.spatial_variables
        meta['cond_variables'] = args.cond_variables
        meta['spatial_shape'] = list(spatial_inputs.shape)
        meta['conditioning_shape'] = list(conditioning.shape)
    else:
        meta['input_variables'] = all_input_variables
        meta['input_shape'] = list(inputs.shape)
    meta['output_shape'] = list(outputs.shape)
    if cloud_filter_meta:
        meta['cloud_filter'] = cloud_filter_meta

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
