#!/usr/bin/env python3
"""
Evaluation for LST Downscaling Model on preprocessed .pt files.

Works on any .pt file (train, test, or validation) that follows the
standard preprocess_data format with 'inputs' and 'outputs' keys.

Loads model checkpoint, runs inference, denormalizes to Kelvin,
computes metrics, and generates comparison figures.

Metrics: ME, MAE, RMSE, Median, STD, RSTD, R², Pearson R,
         error histogram, cold pixel analysis.

Figures:
  - Summary page: scatter density + error histogram
  - Contact sheets: grid of boxes (default 8x4 = 32 per page)
    Each cell shows Target | Prediction | Error side-by-side
    Sorted by RMSE (worst-first) so bad predictions appear on early pages

Usage:
    # Quick terminal check (no figures)
    python evaluate_holdout.py \\
        --model-folder /path/to/checkpoint_best_test_mse \\
        --data-pt /path/to/validation_v8_zscore_rm_coldpattern.pt

    # Full evaluation with all boxes plotted as contact sheets
    python evaluate_holdout.py \\
        --model-folder /path/to/checkpoint_best_test_mse \\
        --data-pt /path/to/validation_v8_zscore_rm_coldpattern.pt \\
        --output-json results/holdout_results.json \\
        --output-figures results/figures/ \\
        --n-samples all

    # Only worst 500 boxes
    python evaluate_holdout.py \\
        --model-folder /path/to/checkpoint_best_test_mse \\
        --data-pt /path/to/validation_v8_zscore_rm_coldpattern.pt \\
        --output-figures results/figures/ \\
        --n-samples 500
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
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.flow_matching_unet import (
    FlowMatchingUNet, flow_matching_sample)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def detect_model_type(model_folder):
    """Read parameters.json to determine model type.

    Returns one of: 'UNET_standard', 'UNET_flow_matching', 'LinearModel'.
    """
    parameters_path = os.path.join(model_folder, 'parameters.json')
    if not os.path.exists(parameters_path):
        raise FileNotFoundError(
            f"No parameters.json found in {model_folder}. "
            f"Is this a valid model checkpoint?")

    with open(parameters_path) as f:
        parameters = json.load(f)

    model_type = parameters.get('type', None)

    if model_type == 'UNET':
        arch = parameters.get('architecture', 'standard')
        if arch == 'flow_matching':
            return 'UNET_flow_matching'
        elif arch == 'conditioned':
            return 'UNET_conditioned'
        else:
            return 'UNET_standard'
    elif model_type == 'LinearModel':
        return 'LinearModel'
    elif model_type is None:
        # Fallback heuristic
        if 'encoded_dim_size' in parameters:
            return 'UNET_standard'
        else:
            return 'LinearModel'
    else:
        raise ValueError(
            f"Unknown model type '{model_type}' in {parameters_path}")


def load_model(model_folder, device):
    """Load model from checkpoint folder. Auto-detects model type.

    Returns:
        (model, model_type) where model_type is one of:
        'UNET_standard', 'UNET_flow_matching', 'LinearModel'.
    """
    model_type = detect_model_type(model_folder)
    print(f"  Detected model type: {model_type}")

    if model_type == 'LinearModel':
        mt = LinearModel()
        mt.load(model_folder)
        mt.weights.to(device)
        mt.weights.eval()
        return mt, model_type

    # --- UNET (both standard and flow_matching) ---
    mt = UNET()
    mt.load(model_folder)

    if model_type == 'UNET_flow_matching':
        # Flow matching uses mt.flow_model, not encoder/decoder
        if mt.flow_model is None:
            input_chan = mt.input_shape[0]
            output_chan = mt.output_shape[0]
            base_ch = getattr(mt, 'base_channels', 64)
            drop = getattr(mt, 'dropout_rate', 0.0)

            print(f"  Rebuilding FlowMatchingUNet: "
                  f"cond={input_chan}, target={output_chan}, "
                  f"base={base_ch}, dropout={drop}")

            mt.flow_model = FlowMatchingUNet(
                cond_channels=input_chan,
                target_channels=output_chan,
                base_channels=base_ch,
                dropout_rate=drop)

            fm_path = os.path.join(model_folder, 'flow_model.weights')
            mt.flow_model.load_state_dict(
                torch.load(fm_path, map_location='cpu', weights_only=False))
            print(f"  Loaded flow_model weights from {fm_path}")

        mt.flow_model.to(device)
        mt.flow_model.eval()

    elif model_type == 'UNET_conditioned':
        from cae_tools.models.conditioned_unet import (
            ConditionedEncoder, ConditionedDecoder)

        if mt.encoder is None or mt.decoder is None:
            # Read conditioned-specific parameters
            parameters_path = os.path.join(model_folder, 'parameters.json')
            with open(parameters_path) as f:
                parameters = json.load(f)

            # Conditioned models store total input channels in input_shape
            # and cond_dim separately. spatial_ch = total - cond_dim.
            cond_dim = parameters.get('cond_dim', 11)
            total_in = mt.input_shape[0]
            spatial_ch = total_in - cond_dim
            output_chan = mt.output_shape[0]
            base_ch = getattr(mt, 'base_channels', 64)
            drop = getattr(mt, 'dropout_rate', 0.0)
            out_act = getattr(mt, 'output_activation', 'none')
            act = parameters.get('activation', 'relu')
            nrb = parameters.get('n_res_blocks_hi', 1)

            print(f"  Rebuilding conditioned encoder/decoder: "
                  f"spatial={spatial_ch}, cond={cond_dim}, "
                  f"out={output_chan}, base={base_ch}, "
                  f"dropout={drop}, output_activation={out_act}, "
                  f"activation={act}, n_res_blocks_hi={nrb}")

            mt.encoder = ConditionedEncoder(
                spatial_in_channels=spatial_ch,
                cond_dim=cond_dim,
                base_channels=base_ch,
                dropout_rate=drop,
                activation=act,
                n_res_blocks_hi=nrb)
            mt.decoder = ConditionedDecoder(
                out_channels=output_chan,
                cond_dim=cond_dim,
                base_channels=base_ch,
                dropout_rate=drop,
                output_activation=out_act,
                activation=act,
                n_res_blocks_hi=nrb)

            enc_path = os.path.join(model_folder, 'encoder.weights')
            dec_path = os.path.join(model_folder, 'decoder.weights')
            mt.encoder.load_state_dict(
                torch.load(enc_path, map_location='cpu', weights_only=False))
            mt.decoder.load_state_dict(
                torch.load(dec_path, map_location='cpu', weights_only=False))
            print(f"  Loaded conditioned encoder/decoder weights")

        mt.encoder.to(device)
        mt.encoder.eval()
        mt.decoder.to(device)
        mt.decoder.eval()

    else:
        # Standard/legacy UNET uses encoder + decoder
        if mt.encoder is None or mt.decoder is None:
            from cae_tools.models.standard_unet import (
                StandardEncoder, StandardDecoder)

            arch = getattr(mt, 'architecture', 'standard')
            input_chan = mt.input_shape[0]
            output_chan = mt.output_shape[0]
            base_ch = getattr(mt, 'base_channels', 64)
            drop = getattr(mt, 'dropout_rate', 0.0)
            out_act = getattr(mt, 'output_activation', 'none')
            act = getattr(mt, 'activation', 'relu')

            print(f"  Rebuilding encoder/decoder: arch={arch}, "
                  f"in={input_chan}, out={output_chan}, "
                  f"base={base_ch}, dropout={drop}, "
                  f"output_activation={out_act}, activation={act}")

            if arch == 'standard':
                mt.encoder = StandardEncoder(
                    in_channels=input_chan,
                    base_channels=base_ch,
                    dropout_rate=drop,
                    activation=act)
                mt.decoder = StandardDecoder(
                    out_channels=output_chan,
                    base_channels=base_ch,
                    dropout_rate=drop,
                    output_activation=out_act,
                    activation=act)
            else:
                raise ValueError(
                    f"Cannot rebuild encoder/decoder for "
                    f"architecture='{arch}'. Only 'standard' is supported "
                    f"by this script.")

            enc_path = os.path.join(model_folder, 'encoder.weights')
            dec_path = os.path.join(model_folder, 'decoder.weights')
            mt.encoder.load_state_dict(
                torch.load(enc_path, map_location='cpu', weights_only=False))
            mt.decoder.load_state_dict(
                torch.load(dec_path, map_location='cpu', weights_only=False))
            print(f"  Loaded encoder/decoder weights from checkpoint")

        mt.encoder.to(device)
        mt.encoder.eval()
        mt.decoder.to(device)
        mt.decoder.eval()

    return mt, model_type


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


def run_inference(mt, model_type, inputs, device, batch_size=256,
                  cond_inputs=None):
    """Run model inference in batches, return normalized predictions.

    Args:
        mt: loaded model (UNET or LinearModel)
        model_type: 'UNET_standard', 'UNET_conditioned',
                    'UNET_flow_matching', or 'LinearModel'
        inputs: tensor of shape (N, C, H, W)
        device: torch device
        batch_size: inference batch size
        cond_inputs: conditioning tensor (N, cond_dim), required for
                     UNET_conditioned, ignored otherwise
    """
    if model_type == 'UNET_conditioned' and cond_inputs is None:
        raise ValueError(
            "cond_inputs is required for UNET_conditioned model type")

    n = inputs.shape[0]
    out_shape = (n, mt.output_shape[0], mt.output_shape[1], mt.output_shape[2])
    predictions = torch.zeros(out_shape, dtype=torch.float32)

    flow_steps = getattr(mt, 'flow_steps', 4)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = inputs[start:end].to(device)

            if model_type == 'UNET_conditioned':
                cond_b = cond_inputs[start:end].to(device)
                encoded, skips = mt.encoder(batch, cond_b)
                pred = mt.decoder(encoded, skips, cond_b)
            elif model_type == 'UNET_standard':
                encoded, skips = mt.encoder(batch)
                pred = mt.decoder(encoded, skips)
            elif model_type == 'UNET_flow_matching':
                B = batch.shape[0]
                target_shape = (B, mt.output_shape[0],
                                batch.shape[2], batch.shape[3])
                pred = flow_matching_sample(
                    mt.flow_model, batch, target_shape,
                    num_steps=flow_steps, device=device)
            else:
                pred = mt.weights(batch)

            predictions[start:end] = pred.cpu()
            if (start + batch_size) % (batch_size * 10) == 0 or end == n:
                print(f"  Inference: {end}/{n} boxes ({100*end//n}%)",
                      flush=True)

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_k, target_k):
    """Compute evaluation metrics in Kelvin space.

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
    box_means = np.nanmean(pred_k.reshape(n_boxes, -1), axis=1)
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


def compute_per_box_metrics(pred_k, target_k):
    """Compute RMSE, MAE, ME, Median Error, Pearson R per box.

    Args:
        pred_k: shape (N, 1, H, W) in Kelvin
        target_k: shape (N, 1, H, W) in Kelvin

    Returns:
        dict with keys 'rmse', 'mae', 'me', 'median', 'pearson_r',
        'mean_pred', 'mean_target', each a numpy array of shape (N,).

    The per-box Pearson R measures spatial pattern fidelity within each
    100x100 pixel box. Unlike the global Pearson R (which is inflated by
    inter-box variance, e.g. cold Scotland vs warm London), the per-box
    Pearson R isolates whether the model reproduces within-box structure
    such as land-cover boundaries, elevation gradients, and urban heat.
    """
    N = pred_k.shape[0]

    # (N, H*W)
    pred_flat = pred_k[:, 0, :, :].reshape(N, -1)
    tgt_flat = target_k[:, 0, :, :].reshape(N, -1)
    err = pred_flat - tgt_flat

    rmse = np.sqrt(np.nanmean(err ** 2, axis=1))
    mae = np.nanmean(np.abs(err), axis=1)
    me = np.nanmean(err, axis=1)
    median = np.nanmedian(err, axis=1)
    mean_pred = np.nanmean(pred_flat, axis=1)
    mean_target = np.nanmean(tgt_flat, axis=1)

    # Per-box Pearson R: correlation between predicted and target pixel
    # values within each box.  For a box where pred or target is spatially
    # constant (zero variance), Pearson R is undefined — we set it to 0.0.
    pred_centered = pred_flat - np.nanmean(pred_flat, axis=1, keepdims=True)
    tgt_centered = tgt_flat - np.nanmean(tgt_flat, axis=1, keepdims=True)

    # Numerator: sum of products of centered values (per box)
    cov_xy = np.nanmean(pred_centered * tgt_centered, axis=1)

    # Denominator: product of standard deviations
    std_pred = np.nanstd(pred_flat, axis=1)
    std_tgt = np.nanstd(tgt_flat, axis=1)
    denom = std_pred * std_tgt

    # Where either field is constant, correlation is undefined → 0.0
    valid_denom = denom > 1e-10
    pearson_r = np.where(valid_denom, cov_xy / denom, 0.0)

    return {
        'rmse': rmse,
        'mae': mae,
        'me': me,
        'median': median,
        'pearson_r': pearson_r,
        'mean_pred': mean_pred,
        'mean_target': mean_target,
    }


# ---------------------------------------------------------------------------
# Figures: contact sheets
# ---------------------------------------------------------------------------

def generate_contact_sheets(pred_k, target_k, box_indices, per_box,
                            output_dir, data_label,
                            rows_per_page=32, dpi=150):
    """Generate contact sheet pages, each showing rows_per_page boxes.

    Each row renders a horizontal triplet: Target | Prediction | Error
    as a (100, 302) composite image using imshow (pixel-perfect rendering).
    The three 100x100 panels are separated by a 1-pixel white line.

    Boxes are drawn in the order given by box_indices (caller controls sort).

    Args:
        pred_k: shape (N, 1, H, W) in Kelvin
        target_k: shape (N, 1, H, W) in Kelvin
        box_indices: 1-D array/list of box indices to plot, in display order
        per_box: dict from compute_per_box_metrics (arrays of shape N)
        output_dir: directory to write PNGs
        data_label: string for titles and filenames
        rows_per_page: number of boxes per page (default: 32)
        dpi: output resolution
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    boxes_per_page = rows_per_page
    n_boxes = len(box_indices)
    n_pages = (n_boxes + boxes_per_page - 1) // boxes_per_page

    H = pred_k.shape[2]  # 100
    W = pred_k.shape[3]  # 100

    # Figure sizing: triplet is 3*W + 2 separator pixels wide, H pixels tall.
    cell_w_in = (3 * W + 2) / dpi
    cell_h_in = H / dpi
    label_w = 1.2   # inches, left margin for row labels
    gap = 0.04       # inches between rows

    fig_w = label_w + cell_w_in + 0.1  # small right margin

    print(f"\nGenerating {n_pages} contact sheet pages "
          f"({boxes_per_page} boxes/page, "
          f"{n_boxes} boxes total) in {output_dir}/")

    t0 = time.time()

    for page_idx in range(n_pages):
        start = page_idx * boxes_per_page
        end = min(start + boxes_per_page, n_boxes)
        page_box_indices = box_indices[start:end]
        n_on_page = len(page_box_indices)

        fig, axes = plt.subplots(
            n_on_page, 1,
            figsize=(fig_w,
                     0.25 + n_on_page * (cell_h_in + gap) + 0.5),
            dpi=dpi,
            squeeze=False
        )

        fig.suptitle(
            f'{data_label}  —  page {page_idx+1}/{n_pages}  '
            f'(boxes {start+1}–{end} of {n_boxes}, sorted worst-first)',
            fontsize=8, fontweight='bold', y=0.995
        )

        for row_idx, bi in enumerate(page_box_indices):
            ax = axes[row_idx, 0]

            tgt = target_k[bi, 0, :, :]   # (H, W)
            prd = pred_k[bi, 0, :, :]     # (H, W)
            err = prd - tgt               # (H, W)

            # Shared vmin/vmax for target and prediction
            vmin_tp = min(float(np.nanmin(tgt)), float(np.nanmin(prd)))
            vmax_tp = max(float(np.nanmax(tgt)), float(np.nanmax(prd)))

            # Symmetric range for error, at least ±1K
            err_abs_max = max(abs(float(np.nanmin(err))),
                              abs(float(np.nanmax(err))),
                              1.0)

            # Normalize arrays to [0, 1] for RGB compositing
            # Target and prediction: shared linear scale
            range_tp = vmax_tp - vmin_tp
            if range_tp < 1e-6:
                range_tp = 1.0
            tgt_norm = np.clip((tgt - vmin_tp) / range_tp, 0, 1)
            prd_norm = np.clip((prd - vmin_tp) / range_tp, 0, 1)

            # Error: symmetric around 0, mapped to [-1, 1]
            err_norm = np.clip(err / err_abs_max, -1, 1)

            # Apply colormaps to get RGB arrays (H, W, 3)
            import matplotlib.cm as cm
            tgt_rgb = cm.inferno(tgt_norm)[:, :, :3]
            prd_rgb = cm.inferno(prd_norm)[:, :, :3]

            # RdBu_r: 0.0 = blue (negative), 0.5 = white (zero), 1.0 = red (positive)
            err_rgb = cm.RdBu_r((err_norm + 1) / 2)[:, :, :3]

            # Separator column (1 pixel wide, white)
            sep = np.ones((H, 1, 3), dtype=np.float64)

            # Composite: target | sep | prediction | sep | error
            composite = np.concatenate([tgt_rgb, sep, prd_rgb, sep, err_rgb],
                                       axis=1)  # (H, 3W+2, 3)

            ax.imshow(composite, aspect='equal', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label: box index + per-box metrics
            rmse_i = per_box['rmse'][bi]
            mae_i = per_box['mae'][bi]
            me_i = per_box['me'][bi]
            median_i = per_box['median'][bi]
            pearson_i = per_box['pearson_r'][bi]
            mean_pred_i = per_box['mean_pred'][bi]
            mean_tgt_i = per_box['mean_target'][bi]
            ax.set_ylabel(
                f'#{bi}  RMSE={rmse_i:.1f}  MAE={mae_i:.1f}  '
                f'ME={me_i:+.1f}  Med={median_i:+.1f}\n'
                f'R={pearson_i:.3f}  '
                f'pred={mean_pred_i:.0f}K  tgt={mean_tgt_i:.0f}K',
                fontsize=5, rotation=0, labelpad=75,
                verticalalignment='center'
            )

        # Column header on the first row's axes
        # Mark the three panels: Target | Prediction | Error
        first_ax = axes[0, 0]
        first_ax.text(W * 0.5, -3, 'Target', fontsize=6,
                      ha='center', va='bottom', transform=first_ax.transData)
        first_ax.text(W * 1.5 + 1, -3, 'Prediction', fontsize=6,
                      ha='center', va='bottom', transform=first_ax.transData)
        first_ax.text(W * 2.5 + 2, -3, 'Error', fontsize=6,
                      ha='center', va='bottom', transform=first_ax.transData)

        plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.01,
                            hspace=0.15)

        fname = os.path.join(
            output_dir,
            f'contact_{data_label}_page{page_idx+1:04d}.png'
        )
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        if (page_idx + 1) % 10 == 0 or page_idx == n_pages - 1:
            elapsed = time.time() - t0
            rate = (page_idx + 1) / elapsed
            remaining = (n_pages - page_idx - 1) / rate if rate > 0 else 0
            print(f"  Pages: {page_idx+1}/{n_pages} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n_pages} pages in {elapsed:.1f}s "
          f"({elapsed/n_pages:.2f}s/page)")


def generate_summary_figure(pred_k, target_k, metrics, output_path,
                            data_label):
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

    # Density scatter (hexbin) — subsample for speed
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
                       f'N={metrics["n_pixels"]:,}\n'
                       f'Box Pearson R: '
                       f'mean={metrics.get("Box_Pearson_R_Mean", 0):.4f}  '
                       f'median={metrics.get("Box_Pearson_R_Median", 0):.4f}  '
                       f'std={metrics.get("Box_Pearson_R_Std", 0):.4f}',
                       fontsize=9)
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

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary figure saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--n-samples", default="all",
                        help="Number of boxes to plot: integer or 'all' "
                             "(default: all)")
    parser.add_argument("--sort-by", default="rmse_desc",
                        choices=["rmse_desc", "rmse_asc",
                                 "mae_desc", "mae_asc",
                                 "me_desc", "me_asc",
                                 "pearson_asc", "pearson_desc",
                                 "index"],
                        help="Sort order for contact sheets "
                             "(default: rmse_desc = worst first). "
                             "pearson_asc = worst spatial correlation first.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic sample selection "
                             "when --n-samples < total boxes. "
                             "Fixed default (42) ensures the same boxes are "
                             "selected across different model evaluations. "
                             "(default: 42)")
    parser.add_argument("--rows-per-page", type=int, default=32,
                        help="Boxes per contact sheet page (default: 32)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu, auto-detected if not set)")
    parser.add_argument("--label", default=None,
                        help="Label for this evaluation run")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for output figures (default: 150)")

    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading data: {args.data_pt}")
    t0 = time.time()
    data = torch.load(args.data_pt, map_location='cpu', weights_only=False)

    # Handle both standard and conditioned .pt formats
    cond_inputs = None
    if 'spatial_inputs' in data:
        # Conditioned format: spatial_inputs + cond_inputs
        spatial_inputs = data['spatial_inputs']
        cond_inputs = data['conditioning']
        print(f"  Conditioned format: spatial {spatial_inputs.shape}, "
              f"cond {cond_inputs.shape}")
        # For conditioned UNet, keep them separate
        inputs = spatial_inputs
    else:
        spatial_inputs = None
        cond_inputs = None
        inputs = data['inputs']

    targets = data['outputs']
    norm_params = data['normalisation_parameters']
    input_variables = data.get('input_variables', [])
    output_activation = data.get('output_activation', 'sigmoid')
    print(f"  Loaded {inputs.shape[0]} boxes in {time.time()-t0:.1f}s")
    print(f"  Input shape: {inputs.shape}, "
          f"Output shape: {targets.shape}")
    print(f"  Output activation: {output_activation}")
    print(f"  Input variables: {input_variables}")

    # ── Label ──
    data_label = args.label
    if data_label is None:
        basename = os.path.basename(args.data_pt)
        data_label = basename.replace('.pt', '').replace('_v8_zscore_rm_coldpattern', '')

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = (torch.device("cuda")
                  if torch.cuda.is_available()
                  else torch.device("cpu"))
    print(f"  Using device: {device}")

    # ── Load model ──
    print(f"\nLoading model: {args.model_folder}")
    mt, model_type = load_model(args.model_folder, device)
    print(f"  Architecture: {mt.architecture if hasattr(mt, 'architecture') else 'unknown'}")
    print(f"  Input shape: {mt.input_shape}, Output shape: {mt.output_shape}")

    # ── Channel check ──
    model_in_channels = mt.input_shape[0]
    data_in_channels = inputs.shape[1]
    if model_type == 'UNET_conditioned':
        cond_dim = getattr(mt, 'cond_dim', 0)
        expected_spatial = model_in_channels - cond_dim
        if data_in_channels != expected_spatial:
            print(f"\n*** CHANNEL MISMATCH: Conditioned model expects "
                  f"{expected_spatial} spatial channels but data has "
                  f"{data_in_channels}. ***")
            sys.exit(1)
        if cond_inputs is not None and cond_inputs.shape[1] != cond_dim:
            print(f"\n*** COND MISMATCH: Model expects cond_dim="
                  f"{cond_dim} but data has {cond_inputs.shape[1]}. ***")
            sys.exit(1)
    else:
        # Non-conditioned model (LinearModel, standard UNet) with conditioned data:
        # broadcast conditioning to spatial and concatenate
        if cond_inputs is not None and model_in_channels != data_in_channels:
            H, W = inputs.shape[2], inputs.shape[3]
            cond_broadcast = cond_inputs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            inputs = torch.cat([inputs, cond_broadcast], dim=1)
            cond_inputs = None  # already merged into inputs
            print(f"  Broadcast conditioning -> {inputs.shape[1]} flat channels")
            data_in_channels = inputs.shape[1]
        if model_in_channels != data_in_channels:
            print(f"\n*** CHANNEL MISMATCH: Model expects "
                  f"{model_in_channels} channels but data has "
                  f"{data_in_channels}. ***")
            sys.exit(1)

    # ── Inference ──
    print(f"\nRunning inference on {inputs.shape[0]} boxes "
          f"(batch_size={args.batch_size})...")
    t0 = time.time()
    pred_norm = run_inference(mt, model_type, inputs, device, args.batch_size,
                             cond_inputs=cond_inputs)
    print(f"  Inference complete in {time.time()-t0:.1f}s")

    # ── Denormalize ──
    print("\nDenormalizing to Kelvin...")
    pred_k = denormalize_output(pred_norm.numpy(), norm_params,
                                output_activation)
    target_k = denormalize_output(targets.numpy(), norm_params,
                                  output_activation)
    print(f"  Prediction range: [{pred_k.min():.1f}K, {pred_k.max():.1f}K]")
    print(f"  Target range:     [{target_k.min():.1f}K, {target_k.max():.1f}K]")

    # ── Filter non-UK pixels (land_cover == 0) ──
    # Land cover class 0 = France, Ireland, Channel Islands — outside UKCEH
    # classification boundary. These are valid land but not UK territory.
    spatial_vars = data.get('spatial_variables', input_variables)
    if 'land_cover' in spatial_vars:
        lc_idx = spatial_vars.index('land_cover')
        if 'spatial_inputs' in data:
            lc_norm = data['spatial_inputs'][:, lc_idx].numpy()
        else:
            lc_norm = data['inputs'][:, lc_idx].numpy()
        lc_mn = norm_params['min_inputs']['land_cover']
        lc_mx = norm_params['max_inputs']['land_cover']
        lc_phys = lc_norm * (lc_mx - lc_mn) + lc_mn
        non_uk = (np.round(lc_phys).astype(int) == 0)
        # non_uk shape: (N, 100, 100), pred_k shape: (N, 1, 100, 100)
        non_uk_4d = non_uk[:, np.newaxis, :, :]
        n_non_uk = int(non_uk_4d.sum())
        n_total = int(np.prod(pred_k.shape))
        print(f"\n  Non-UK pixels (land_cover=0): {n_non_uk:,} / {n_total:,} "
              f"({100*n_non_uk/n_total:.1f}%)")
        pred_k = np.where(non_uk_4d, np.nan, pred_k)
        target_k = np.where(non_uk_4d, np.nan, target_k)
        print(f"  Set to NaN — excluded from all metrics")

    # ── Global metrics ──
    print("\nComputing global metrics...")
    metrics = compute_metrics(pred_k, target_k)

    # ── Per-box metrics (needed for sorting, labels, and box Pearson stats) ──
    print("Computing per-box metrics...")
    per_box = compute_per_box_metrics(pred_k, target_k)

    # Add box-level Pearson R aggregates to global metrics.
    # These measure spatial pattern fidelity per 10km box, unlike the
    # global Pearson R which is inflated by inter-box temperature variance.
    metrics['Box_Pearson_R_Mean'] = float(np.mean(per_box['pearson_r']))
    metrics['Box_Pearson_R_Median'] = float(np.median(per_box['pearson_r']))
    metrics['Box_Pearson_R_Std'] = float(np.std(per_box['pearson_r']))
    metrics['Box_Pearson_R_P5'] = float(np.percentile(per_box['pearson_r'], 5))
    metrics['Box_Pearson_R_P25'] = float(np.percentile(per_box['pearson_r'], 25))

    # ── Print summary ──
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
R² (global):    {m['R_squared']:.6f}
Pearson R (global):  {m['Pearson_R']:.6f}
--------------------------------------------
Box Pearson R (spatial pattern fidelity):
  Mean:         {m['Box_Pearson_R_Mean']:.6f}
  Median:       {m['Box_Pearson_R_Median']:.6f}
  Std:          {m['Box_Pearson_R_Std']:.6f}
  P5:           {m['Box_Pearson_R_P5']:.6f}
  P25:          {m['Box_Pearson_R_P25']:.6f}
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

    # ── Build results dict ──
    results = {
        "model": args.model_folder,
        "model_type": model_type,
        "data": args.data_pt,
        "label": data_label,
        "n_boxes": int(inputs.shape[0]),
        "device": str(device),
        "output_activation": output_activation,
        "input_variables": input_variables,
        "metrics": metrics,
    }

    # ── Save JSON ──
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results JSON saved to {args.output_json}")

    # ── Save text ──
    if args.output_text:
        os.makedirs(os.path.dirname(args.output_text) or '.', exist_ok=True)
        with open(args.output_text, 'w') as f:
            f.write(summary)
        print(f"Text summary saved to {args.output_text}")

    # ── Generate figures ──
    if args.output_figures:
        fig_dir = args.output_figures

        # Summary figure (scatter + histogram)
        summary_path = os.path.join(fig_dir, f'summary_{data_label}.png')
        generate_summary_figure(pred_k, target_k, metrics,
                                summary_path, data_label)

        # Determine which boxes to plot
        n_total = pred_k.shape[0]
        if args.n_samples == 'all':
            n_to_plot = n_total
            # When plotting all boxes, no random selection needed
            selected_indices = np.arange(n_total)
        else:
            n_to_plot = min(int(args.n_samples), n_total)
            if n_to_plot == n_total:
                selected_indices = np.arange(n_total)
            else:
                # Deterministic random selection with fixed seed.
                # The seed is fixed (default=42) so that the SAME boxes
                # are selected regardless of which model is being evaluated.
                # This enables apples-to-apples visual comparison across models.
                rng = np.random.default_rng(args.seed)
                selected_indices = rng.choice(n_total, n_to_plot, replace=False)
                print(f"\n  Sample selection: {n_to_plot} of {n_total} boxes "
                      f"(seed={args.seed})")

        # Sort the selected boxes for display order.
        # Sorting is applied AFTER selection, so it only affects page ordering,
        # not which boxes are chosen.
        if args.sort_by == 'rmse_desc':
            display_order = np.argsort(-per_box['rmse'][selected_indices])
        elif args.sort_by == 'rmse_asc':
            display_order = np.argsort(per_box['rmse'][selected_indices])
        elif args.sort_by == 'mae_desc':
            display_order = np.argsort(-per_box['mae'][selected_indices])
        elif args.sort_by == 'mae_asc':
            display_order = np.argsort(per_box['mae'][selected_indices])
        elif args.sort_by == 'me_desc':
            display_order = np.argsort(-np.abs(per_box['me'][selected_indices]))
        elif args.sort_by == 'me_asc':
            display_order = np.argsort(np.abs(per_box['me'][selected_indices]))
        elif args.sort_by == 'pearson_asc':
            display_order = np.argsort(per_box['pearson_r'][selected_indices])
        elif args.sort_by == 'pearson_desc':
            display_order = np.argsort(-per_box['pearson_r'][selected_indices])
        else:  # 'index'
            display_order = np.argsort(selected_indices)

        box_indices = selected_indices[display_order]

        # Print sort-order stats for the selected boxes
        sel_rmse = per_box['rmse'][box_indices]
        print(f"\nSelected {n_to_plot} boxes (sort={args.sort_by}):")
        print(f"  RMSE range: [{sel_rmse.min():.2f}, {sel_rmse.max():.2f}] K")
        print(f"  RMSE median: {np.median(sel_rmse):.2f} K")

        # Estimate output size
        # Each page: ~200-400 KB PNG.
        n_pages = (n_to_plot + args.rows_per_page - 1) // args.rows_per_page
        est_mb = n_pages * 0.3  # ~300 KB per page
        print(f"  Estimated output: {n_pages} pages, ~{est_mb:.0f} MB")

        generate_contact_sheets(
            pred_k, target_k, box_indices, per_box,
            fig_dir, data_label,
            rows_per_page=args.rows_per_page,
            dpi=args.dpi
        )


if __name__ == '__main__':
    main()
