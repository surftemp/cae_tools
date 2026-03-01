#!/usr/bin/env python3
"""
Benchmark baselines for LST downscaling.

Trains three baseline models and reports metrics for comparison against UNet:

1. ERA5-only: no training. Copies ERA5 coarse temperature channel to output.
   Measures how much improvement ANY model provides over raw ERA5.

2. Pixel-linear: Conv2d(11, 1, 1). Per-pixel linear regression from 11 input
   channels. 12 parameters. Measures linearly accessible signal in inputs.

3. Pixel-MLP: stacked 1x1 convolutions (11→64→32→1 with ReLU). Per-pixel
   nonlinear regression. ~2,881 parameters. Measures nonlinear per-pixel
   signal without spatial context.

The gap between these tells you where remaining signal lives:
  ERA5 → linear:  high-res static channels (elevation, land cover) help
  linear → MLP:   nonlinear channel interactions exist
  MLP → UNet:     spatial context contributes

All models saved in LinearModel-compatible format for use with apply_cae
and the full scoring pipeline.

Usage:
    python run_benchmarks.py \\
        --train-pt /path/to/train.pt \\
        --test-pt /path/to/test.pt \\
        --output-dir /path/to/benchmark_models \\
        --nr-epochs 500 \\
        --batch-size 64
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add cae_tools to path if running standalone
try:
    from cae_tools.models.preprocessed_dataset import PreprocessedDataset
    from cae_tools.models.linear_model import LinearModel
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from cae_tools.models.preprocessed_dataset import PreprocessedDataset
    from cae_tools.models.linear_model import LinearModel


ERA5_CHANNEL_IDX = 3


def compute_era5_baseline(test_ds, norm_params):
    """Compute ERA5-only baseline: use ERA5 channel as prediction.

    ERA5 is normalised with ERA5 min/max. Output is normalised with
    LST min/max. So we denormalise both to Kelvin and compare.

    Returns: dict with mse, rmse, mae in Kelvin.
    """
    if isinstance(norm_params, list):
        min_out = list(norm_params[2].values())[0]
        max_out = list(norm_params[3].values())[0]
        era5_min = norm_params[0]['era5_skt']
        era5_max = norm_params[1]['era5_skt']
    else:
        min_out = norm_params.get('min_output', 0)
        max_out = norm_params.get('max_output', 1)
        era5_min = norm_params['min_inputs']['era5_skt']
        era5_max = norm_params['max_inputs']['era5_skt']

    out_range = max_out - min_out
    era5_range = era5_max - era5_min

    se_sum = 0.0
    ae_sum = 0.0
    n_pixels = 0

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    for inputs, targets, _ in loader:
        # Denormalise to Kelvin
        era5_k = inputs[:, ERA5_CHANNEL_IDX:ERA5_CHANNEL_IDX + 1, :, :] * era5_range + era5_min
        lst_k = targets * out_range + min_out

        diff = era5_k - lst_k
        se_sum += (diff ** 2).sum().item()
        ae_sum += diff.abs().sum().item()
        n_pixels += diff.numel()

    mse = se_sum / n_pixels
    mae = ae_sum / n_pixels
    rmse = mse ** 0.5
    return {'mse_kelvin': mse, 'rmse_kelvin': rmse, 'mae_kelvin': mae}


def compute_model_test_metrics(model_weights, test_ds, norm_params, device):
    """Compute test metrics for a trained model in both normalised and Kelvin space."""
    if isinstance(norm_params, list):
        min_out = list(norm_params[2].values())[0]
        max_out = list(norm_params[3].values())[0]
    else:
        min_out = norm_params.get('min_output', 0)
        max_out = norm_params.get('max_output', 1)
    out_range = max_out - min_out

    model_weights.eval()
    mse_sum = 0.0
    ae_sum = 0.0
    n_pixels = 0

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred = model_weights(inputs)

            diff = pred - targets
            mse_sum += (diff ** 2).sum().item()
            ae_sum += diff.abs().sum().item()
            n_pixels += diff.numel()

    mse_norm = mse_sum / n_pixels
    mae_norm = ae_sum / n_pixels
    rmse_k = mse_norm ** 0.5 * out_range
    mae_k = mae_norm * out_range

    return {
        'mse_norm': mse_norm,
        'rmse_kelvin': rmse_k,
        'mae_kelvin': mae_k,
    }


def train_benchmark(architecture, train_ds, test_ds, output_dir,
                    nr_epochs, batch_size, lr):
    """Train one benchmark model and save it."""
    model_dir = os.path.join(output_dir, f'benchmark_{architecture}')

    mt = LinearModel(
        batch_size=batch_size,
        nr_epochs=nr_epochs,
        lr=lr,
        architecture=architecture,
        test_interval=10,
    )

    mt.train_from_datasets(
        train_ds, test_ds,
        model_path=model_dir,
        training_paths="benchmark",
        testing_paths="benchmark",
    )

    return mt, model_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train benchmark baselines for LST downscaling")
    parser.add_argument("--train-pt", required=True,
                        help="Path to training .pt file")
    parser.add_argument("--test-pt", required=True,
                        help="Path to test .pt file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save benchmark models")
    parser.add_argument("--nr-epochs", type=int, default=500,
                        help="Training epochs (default 500)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default 64)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default 0.001)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LST Downscaling Benchmarks")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_ds = PreprocessedDataset(args.train_pt)
    test_ds = PreprocessedDataset(args.test_pt)
    test_ds.set_normalisation_parameters(train_ds.get_normalisation_parameters())
    norm_params = train_ds.get_normalisation_parameters()

    results = {}

    # ---- Benchmark 1: ERA5 baseline ----
    print("\n" + "=" * 60)
    print("Benchmark 1: ERA5-only (no model, raw ERA5 broadcast)")
    print("=" * 60)
    era5_metrics = compute_era5_baseline(test_ds, norm_params)
    results['era5_only'] = era5_metrics
    print(f"  RMSE: {era5_metrics['rmse_kelvin']:.2f}K")
    print(f"  MAE:  {era5_metrics['mae_kelvin']:.2f}K")

    # ---- Benchmark 2: Pixel-linear ----
    print("\n" + "=" * 60)
    print("Benchmark 2: Pixel-linear (Conv2d 11→1, kernel_size=1)")
    print("=" * 60)
    mt_linear, dir_linear = train_benchmark(
        'pixel_linear', train_ds, test_ds, args.output_dir,
        args.nr_epochs, args.batch_size, args.lr)

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    linear_metrics = compute_model_test_metrics(
        mt_linear.weights, test_ds, norm_params, device)
    results['pixel_linear'] = linear_metrics
    print(f"\n  Final test RMSE: {linear_metrics['rmse_kelvin']:.2f}K")
    print(f"  Final test MAE:  {linear_metrics['mae_kelvin']:.2f}K")
    print(f"  Saved to: {dir_linear}")

    # ---- Benchmark 3: Pixel-MLP ----
    print("\n" + "=" * 60)
    print("Benchmark 3: Pixel-MLP (Conv2d 11→64→32→1, kernel_size=1)")
    print("=" * 60)
    mt_mlp, dir_mlp = train_benchmark(
        'pixel_mlp', train_ds, test_ds, args.output_dir,
        args.nr_epochs, args.batch_size, args.lr)

    mlp_metrics = compute_model_test_metrics(
        mt_mlp.weights, test_ds, norm_params, device)
    results['pixel_mlp'] = mlp_metrics
    print(f"\n  Final test RMSE: {mlp_metrics['rmse_kelvin']:.2f}K")
    print(f"  Final test MAE:  {mlp_metrics['mae_kelvin']:.2f}K")
    print(f"  Saved to: {dir_mlp}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'RMSE (K)':<12} {'MAE (K)':<12} {'Parameters':<12}")
    print("-" * 56)
    print(f"{'ERA5-only':<20} {era5_metrics['rmse_kelvin']:<12.2f} "
          f"{era5_metrics['mae_kelvin']:<12.2f} {'0':>12}")

    n_linear = sum(p.numel() for p in mt_linear.weights.parameters())
    print(f"{'Pixel-linear':<20} {linear_metrics['rmse_kelvin']:<12.2f} "
          f"{linear_metrics['mae_kelvin']:<12.2f} {n_linear:>12,}")

    n_mlp = sum(p.numel() for p in mt_mlp.weights.parameters())
    print(f"{'Pixel-MLP':<20} {mlp_metrics['rmse_kelvin']:<12.2f} "
          f"{mlp_metrics['mae_kelvin']:<12.2f} {n_mlp:>12,}")

    print(f"\n{'UNet (for reference)':<20} {'~2.0-2.5':<12} {'~1.8-2.0':<12} "
          f"{'~14,000,000':>12}")

    print("\nGap analysis:")
    print(f"  ERA5 → Linear:  {era5_metrics['rmse_kelvin'] - linear_metrics['rmse_kelvin']:.2f}K "
          f"(static high-res channels)")
    print(f"  Linear → MLP:   {linear_metrics['rmse_kelvin'] - mlp_metrics['rmse_kelvin']:.2f}K "
          f"(nonlinear channel interactions)")
    print(f"  MLP → UNet:     compare with your UNet test RMSE "
          f"(spatial context contribution)")

    # Save summary
    results_path = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
