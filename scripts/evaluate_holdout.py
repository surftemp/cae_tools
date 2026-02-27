#!/usr/bin/env python3
"""
Evaluation for LST Downscaling Model on preprocessed .pt files.

Works on any .pt file (train, test, or validation) that follows the
standard preprocess_data format with 'inputs' and 'outputs' keys.

Loads model checkpoint, runs inference, denormalizes to Kelvin,
computes metrics, and optionally generates comparison figures.

Metrics: ME, MAE, RMSE, Median, STD, RSTD, R², Pearson R,
         error histogram, cold pixel analysis.

Figures: N evenly-spaced sample boxes showing Target | Prediction | Error
         with aligned colorbars (pcolormesh, no interpolation).

Usage:
    # Quick terminal check (no file output)
    python evaluate_holdout.py \\
        --model-folder /path/to/checkpoint_best_test_mse \\
        --data-pt /path/to/validation_v8_zscore_rm_coldpattern.pt

    # Full evaluation with figures and JSON
    python evaluate_holdout.py \\
        --model-folder /path/to/checkpoint_best_test_mse \\
        --data-pt /path/to/validation_v8_zscore_rm_coldpattern.pt \\
        --output-json results/holdout_results.json \\
        --output-figures results/figures/ \\
        --n-samples 24
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Add cae_tools to path if not installed
sys.path.insert(0, '/home/users/shaerdan/cae_tools_pB/src')

from cae_tools.models.unet import UNET


def load_model(model_folder, device):
    """Load UNET model from checkpoint folder."""
    mt = UNET()
    mt.load(model_folder)

    if mt.encoder is not None:
        mt.encoder.to(device)
        mt.encoder.eval()
    if mt.decoder is not None:
        mt.decoder.to(device)
        mt.decoder.eval()

    return mt


def denormalize_output(arr, norm_params, output_activation='sigmoid'):
    """Convert normalized output back to Kelvin.

    sigmoid mode: physical = arr * (max - min) + min   [0,1] -> K
    tanh mode:    physical = ((arr + 1) / 2) * (max - min) + min  [-1,1] -> K
    """
    if isinstance(norm_params, dict):
        min_out = norm_params['min_output']
        max_out = norm_params['max_output']
    else:
        min_out = list(norm_params[2].values())[0] if isinstance(norm_params[2], dict) else norm_params[2]
        max_out = list(norm_params[3].values())[0] if isinstance(norm_params[3], dict) else norm_params[3]

    range_out = max_out - min_out
    if output_activation == 'tanh':
        return ((arr + 1.0) / 2.0) * range_out + min_out
    else:
        return arr * range_out + min_out


def run_inference(mt, inputs, device, batch_size=256):
    """Run model inference in batches, return normalized predictions."""
    n = inputs.shape[0]
    predictions = torch.zeros(n, 1, inputs.shape[2], inputs.shape[3],
                              dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = inputs[start:end].to(device)

            if mt.architecture == 'flow_matching' and mt.flow_model is not None:
                from cae_tools.models.flow_matching_unet import flow_matching_sample
                pred = flow_matching_sample(
                    mt.flow_model, batch, mt.output_shape, mt.flow_steps, device
                )
            else:
                encoded, skip = mt.encoder(batch)
                pred = mt.decoder(encoded, skip)

            predictions[start:end] = pred.cpu()

            if (start // batch_size) % 10 == 0:
                print(f"  Inference: {end}/{n} boxes ({100*end/n:.0f}%)",
                      flush=True)

    return predictions


def compute_metrics(pred_k, target_k):
    """Compute comprehensive evaluation metrics.

    Args:
        pred_k: predictions in Kelvin, shape (N, 1, H, W)
        target_k: targets in Kelvin, shape (N, 1, H, W)

    Returns:
        dict of metrics
    """
    pred_flat = pred_k.ravel()
    tgt_flat = target_k.ravel()

    valid = ~np.isnan(pred_flat) & ~np.isnan(tgt_flat)
    pred = pred_flat[valid]
    tgt = tgt_flat[valid]
    n_pixels = len(pred)

    if n_pixels == 0:
        return {"error": "No valid pixels found"}

    error = pred - tgt

    me = float(np.mean(error))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    median_error = float(np.median(error))
    std_error = float(np.std(error))

    q25, q75 = np.percentile(error, [25, 75])
    rstd = float((q75 - q25) / 1.349)

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((tgt - np.mean(tgt)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    r_corr = float(np.corrcoef(pred, tgt)[0, 1])

    p5, p95 = np.percentile(error, [5, 95])

    bin_edges = np.arange(-20.25, 20.75, 0.5)
    hist_counts, hist_edges = np.histogram(error, bins=bin_edges)
    histogram = {
        f"{hist_edges[i]:.1f}_to_{hist_edges[i+1]:.1f}": int(hist_counts[i])
        for i in range(len(hist_counts))
        if hist_counts[i] > 0
    }

    n_cold = int(np.sum(pred < 280))
    pct_cold = float(100 * n_cold / n_pixels)

    n_boxes = pred_k.shape[0]
    box_means = pred_k.reshape(n_boxes, -1).mean(axis=1)
    n_cold_boxes = int(np.sum(box_means < 280))

    return {
        "n_pixels": n_pixels,
        "n_boxes": n_boxes,
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "Median_Error": median_error,
        "STD_Error": std_error,
        "RSTD_Error": rstd,
        "R_squared": r_squared,
        "Pearson_R": r_corr,
        "P5_Error": float(p5),
        "P95_Error": float(p95),
        "Q25_Error": float(q25),
        "Q75_Error": float(q75),
        "Mean_Prediction_K": float(np.mean(pred)),
        "Mean_Target_K": float(np.mean(tgt)),
        "Pred_Range_K": [float(np.min(pred)), float(np.max(pred))],
        "Target_Range_K": [float(np.min(tgt)), float(np.max(tgt))],
        "Cold_Pixels_lt280K": n_cold,
        "Cold_Pixel_Pct": pct_cold,
        "Cold_Boxes_lt280K": n_cold_boxes,
        "Cold_Box_Pct": float(100 * n_cold_boxes / n_boxes) if n_boxes > 0 else 0,
        "Error_Histogram_0.5K": histogram,
    }


def select_sample_indices(n_total, n_samples):
    """Select n_samples indices spread evenly across the dataset.

    If boxes are chronologically ordered (as preprocess_data produces),
    this naturally spreads samples across months.
    """
    if n_samples >= n_total:
        return list(range(n_total))
    return [int(round(i * (n_total - 1) / (n_samples - 1)))
            for i in range(n_samples)]


def generate_comparison_figures(pred_k, target_k, sample_indices,
                                output_dir, data_label):
    """Generate Target | Prediction | Error comparison figures.

    Each figure shows one box as a 1x3 panel:
      Left:   Landsat target (K)
      Middle: Model prediction (K)
      Right:  Error = Prediction - Target (K)

    Target and Prediction share the same colorbar range.
    Error has a symmetric diverging colorbar centered on zero.
    All panels use pcolormesh with no interpolation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    n_figs = len(sample_indices)
    print(f"\nGenerating {n_figs} comparison figures in {output_dir}/")

    for fig_idx, box_idx in enumerate(sample_indices):
        tgt = target_k[box_idx, 0, :, :]
        prd = pred_k[box_idx, 0, :, :]
        err = prd - tgt

        # Shared vmin/vmax covering both target and prediction
        vmin_tp = min(float(np.nanmin(tgt)), float(np.nanmin(prd)))
        vmax_tp = max(float(np.nanmax(tgt)), float(np.nanmax(prd)))

        # Symmetric range for error panel
        err_abs_max = max(abs(float(np.nanmin(err))),
                          abs(float(np.nanmax(err))))
        err_abs_max = max(err_abs_max, 0.5)  # at least ±0.5K

        # Per-box metrics
        box_me = float(np.nanmean(err))
        box_mae = float(np.nanmean(np.abs(err)))
        box_rmse = float(np.sqrt(np.nanmean(err ** 2)))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                                 constrained_layout=True)

        # Target
        im0 = axes[0].pcolormesh(tgt, vmin=vmin_tp, vmax=vmax_tp,
                                  cmap='inferno', shading='nearest')
        axes[0].set_title(f'Target (Landsat)\n'
                          f'mean={np.nanmean(tgt):.1f}K',
                          fontsize=10)
        axes[0].set_aspect('equal')
        axes[0].invert_yaxis()
        plt.colorbar(im0, ax=axes[0], label='K', shrink=0.8)

        # Prediction
        im1 = axes[1].pcolormesh(prd, vmin=vmin_tp, vmax=vmax_tp,
                                  cmap='inferno', shading='nearest')
        axes[1].set_title(f'Prediction (Model)\n'
                          f'mean={np.nanmean(prd):.1f}K',
                          fontsize=10)
        axes[1].set_aspect('equal')
        axes[1].invert_yaxis()
        plt.colorbar(im1, ax=axes[1], label='K', shrink=0.8)

        # Error
        im2 = axes[2].pcolormesh(err, vmin=-err_abs_max, vmax=err_abs_max,
                                  cmap='RdBu_r', shading='nearest')
        axes[2].set_title(f'Error (Pred \u2212 Target)\n'
                          f'ME={box_me:+.2f}K  MAE={box_mae:.2f}K  '
                          f'RMSE={box_rmse:.2f}K',
                          fontsize=10)
        axes[2].set_aspect('equal')
        axes[2].invert_yaxis()
        plt.colorbar(im2, ax=axes[2], label='K', shrink=0.8)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'{data_label}  \u2014  Box {box_idx} '
                     f'(sample {fig_idx+1}/{n_figs})',
                     fontsize=12, fontweight='bold')

        fname = os.path.join(output_dir,
                             f'comparison_{data_label}_box{box_idx:05d}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (fig_idx + 1) % 8 == 0 or fig_idx == n_figs - 1:
            print(f"  Figures: {fig_idx+1}/{n_figs}", flush=True)

    print(f"  Saved {n_figs} comparison figures to {output_dir}/")


def generate_summary_figure(pred_k, target_k, metrics, output_path, data_label):
    """Generate a single-page summary: scatter density + error histogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pred_flat = pred_k.ravel()
    tgt_flat = target_k.ravel()
    valid = ~np.isnan(pred_flat) & ~np.isnan(tgt_flat)
    pred = pred_flat[valid]
    tgt = tgt_flat[valid]
    error = pred - tgt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Density scatter (hexbin)
    n_plot = min(len(pred), 500_000)
    if n_plot < len(pred):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pred), n_plot, replace=False)
        p_plot, t_plot = pred[idx], tgt[idx]
    else:
        p_plot, t_plot = pred, tgt

    axes[0].hexbin(t_plot, p_plot, gridsize=150, cmap='inferno',
                    mincnt=1, linewidths=0.1)
    lo = min(t_plot.min(), p_plot.min())
    hi = max(t_plot.max(), p_plot.max())
    axes[0].plot([lo, hi], [lo, hi], 'w--', linewidth=1, alpha=0.8)
    axes[0].set_xlabel('Target (K)')
    axes[0].set_ylabel('Prediction (K)')
    axes[0].set_title(f'{data_label}\n'
                       f'R\u00b2={metrics["R_squared"]:.4f}  '
                       f'RMSE={metrics["RMSE"]:.3f}K  '
                       f'MAE={metrics["MAE"]:.3f}K  '
                       f'N={metrics["n_pixels"]:,}')
    axes[0].set_aspect('equal')

    # Error histogram
    bin_edges = np.arange(-15, 15.5, 0.25)
    axes[1].hist(error, bins=bin_edges, color='steelblue',
                  edgecolor='none', alpha=0.85)
    axes[1].axvline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].axvline(metrics['ME'], color='red', linewidth=1.2,
                     label=f'ME = {metrics["ME"]:+.3f}K')
    axes[1].axvline(metrics['Median_Error'], color='orange', linewidth=1.2,
                     label=f'Median = {metrics["Median_Error"]:+.3f}K')
    axes[1].set_xlabel('Error (K)')
    axes[1].set_ylabel('Pixel count')
    axes[1].set_title(f'Error Distribution\n'
                       f'STD={metrics["STD_Error"]:.3f}K  '
                       f'RSTD={metrics["RSTD_Error"]:.3f}K')
    axes[1].legend(fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LST downscaling model on preprocessed .pt data "
                    "(works on train, test, or validation .pt files)")
    parser.add_argument("--model-folder", required=True,
                        help="Path to model checkpoint folder")
    parser.add_argument("--data-pt", required=True,
                        help="Path to .pt file (train/test/validation)")
    parser.add_argument("--output-json", default=None,
                        help="Path to save results JSON")
    parser.add_argument("--output-text", default=None,
                        help="Path to save human-readable results")
    parser.add_argument("--output-figures", default=None,
                        help="Directory to save comparison figures")
    parser.add_argument("--n-samples", type=int, default=24,
                        help="Number of sample comparison figures "
                             "(default: 24, spread evenly across dataset)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu, auto-detected if not set)")
    parser.add_argument("--label", default=None,
                        help="Label for figures "
                             "(auto-derived from filename if not set)")
    args = parser.parse_args()

    # Auto-derive label from filename
    if args.label:
        data_label = args.label
    else:
        data_label = os.path.splitext(os.path.basename(args.data_pt))[0]

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data: {args.data_pt}")
    t0 = time.time()
    val_data = torch.load(args.data_pt, map_location="cpu")
    inputs = val_data['inputs']
    targets = val_data['outputs']
    norm_params = val_data['normalisation_parameters']
    output_activation = val_data.get('output_activation', 'sigmoid')
    input_variables = val_data.get('input_variables', [])
    print(f"  Loaded {inputs.shape[0]} boxes in {time.time()-t0:.1f}s")
    print(f"  Input shape: {inputs.shape}, Output shape: {targets.shape}")
    print(f"  Output activation: {output_activation}")
    print(f"  Input variables: {input_variables}")

    # Load model
    print(f"\nLoading model: {args.model_folder}")
    mt = load_model(args.model_folder, device)
    print(f"  Architecture: {mt.architecture}")
    print(f"  Input shape: {mt.input_shape}, Output shape: {mt.output_shape}")

    # Verify channel count compatibility
    model_in_channels = mt.input_shape[0] if mt.input_shape else None
    data_in_channels = inputs.shape[1]
    if model_in_channels and model_in_channels != data_in_channels:
        print(f"\n*** FATAL: Channel mismatch! Model expects "
              f"{model_in_channels} channels but data has "
              f"{data_in_channels}. ***")
        sys.exit(1)

    # Run inference
    print(f"\nRunning inference on {inputs.shape[0]} boxes "
          f"(batch_size={args.batch_size})...")
    t0 = time.time()
    pred_norm = run_inference(mt, inputs, device, args.batch_size)
    print(f"  Inference complete in {time.time()-t0:.1f}s")

    # Denormalize to Kelvin
    print("\nDenormalizing to Kelvin...")
    pred_k = denormalize_output(pred_norm.numpy(), norm_params,
                                output_activation)
    target_k = denormalize_output(targets.numpy(), norm_params,
                                  output_activation)
    print(f"  Prediction range: [{pred_k.min():.1f}K, {pred_k.max():.1f}K]")
    print(f"  Target range:     [{target_k.min():.1f}K, {target_k.max():.1f}K]")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(pred_k, target_k)

    # Build results
    results = {
        "model": args.model_folder,
        "data": args.data_pt,
        "label": data_label,
        "n_boxes": int(inputs.shape[0]),
        "device": str(device),
        "output_activation": output_activation,
        "input_variables": input_variables,
        "metrics": metrics,
    }

    # Print summary
    m = metrics
    summary = f"""
============================================
EVALUATION RESULTS: {data_label}
============================================
Model:      {os.path.basename(os.path.dirname(args.model_folder))}/{os.path.basename(args.model_folder)}
Data:       {os.path.basename(args.data_pt)}
Boxes:      {m['n_boxes']}
Pixels:     {m['n_pixels']:,}
Device:     {device}
--------------------------------------------
ME  (bias):     {m['ME']:+.4f} K
MAE:            {m['MAE']:.4f} K
RMSE:           {m['RMSE']:.4f} K
Median Error:   {m['Median_Error']:+.4f} K
STD(error):     {m['STD_Error']:.4f} K
RSTD(error):    {m['RSTD_Error']:.4f} K
R²:             {m['R_squared']:.6f}
Pearson R:      {m['Pearson_R']:.6f}
--------------------------------------------
P5  error:      {m['P5_Error']:+.3f} K
P25 error:      {m['Q25_Error']:+.3f} K
P75 error:      {m['Q75_Error']:+.3f} K
P95 error:      {m['P95_Error']:+.3f} K
--------------------------------------------
Mean pred:      {m['Mean_Prediction_K']:.2f} K
Mean target:    {m['Mean_Target_K']:.2f} K
Pred range:     [{m['Pred_Range_K'][0]:.1f}, {m['Pred_Range_K'][1]:.1f}] K
Target range:   [{m['Target_Range_K'][0]:.1f}, {m['Target_Range_K'][1]:.1f}] K
--------------------------------------------
Cold pixels (<280K):  {m['Cold_Pixels_lt280K']:,} ({m['Cold_Pixel_Pct']:.2f}%)
Cold boxes (<280K):   {m['Cold_Boxes_lt280K']} / {m['n_boxes']} ({m['Cold_Box_Pct']:.1f}%)
============================================
"""
    print(summary)

    # Save JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results JSON saved to {args.output_json}")

    # Save text
    if args.output_text:
        os.makedirs(os.path.dirname(args.output_text) or '.', exist_ok=True)
        with open(args.output_text, 'w') as f:
            f.write(summary)
        print(f"Text summary saved to {args.output_text}")

    # Generate figures
    if args.output_figures:
        fig_dir = args.output_figures

        # Summary figure (scatter + histogram)
        summary_path = os.path.join(fig_dir, f'summary_{data_label}.png')
        os.makedirs(fig_dir, exist_ok=True)
        generate_summary_figure(pred_k, target_k, metrics,
                                summary_path, data_label)

        # Per-box comparison figures
        sample_indices = select_sample_indices(inputs.shape[0],
                                               args.n_samples)
        generate_comparison_figures(pred_k, target_k, sample_indices,
                                    fig_dir, data_label)


if __name__ == '__main__':
    main()
