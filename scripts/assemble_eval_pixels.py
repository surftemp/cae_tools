"""
Per-pixel evaluation from test .pt + model checkpoint.
Supports both legacy (standard UNet) and conditioned formats.

Auto-detects format from the .pt file:
  - Legacy: 'inputs' key, all channels broadcast to (N, C, H, W)
  - Conditioned: 'spatial_inputs' + 'conditioning' keys

Usage:
  python assemble_eval_pixels.py \
      --test-pt /path/to/test.pt \
      --model-folder /path/to/checkpoint \
      --output-dir /path/to/eval_output \
      --model-id my_model
"""

import argparse
import os
import json
import sys
import time

import numpy as np
import torch
import pandas as pd


def denorm_value(val, var_name, norm_params):
    mn = norm_params['min_inputs'][var_name]
    mx = norm_params['max_inputs'][var_name]
    return val * (mx - mn) + mn


def denorm_output(val, norm_params):
    mn = norm_params['min_output']
    mx = norm_params['max_output']
    activation = norm_params.get('output_activation', 'sigmoid')
    if activation == 'tanh':
        return mn + ((val + 1) / 2) * (mx - mn)
    else:
        return mn + val * (mx - mn)


def recover_doy_and_month(sin_phys, cos_phys):
    d1 = np.arcsin(np.clip(sin_phys, 0, 1)) * 366 / np.pi
    d2 = 366 - d1
    c1 = np.cos((d1 / 366) * 2 * np.pi)
    c2 = np.cos((d2 / 366) * 2 * np.pi)
    doy = np.where(np.abs(c1 - cos_phys) < np.abs(c2 - cos_phys), d1, d2)
    doy = np.clip(doy, 1, 366).astype(int)

    boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]
    months = np.zeros(len(doy), dtype=int)
    for m in range(12):
        mask = (doy > boundaries[m]) & (doy <= boundaries[m + 1])
        months[mask] = m + 1
    months[months == 0] = 1
    return doy, months


def compute_stats(df, group_col, group_vals=None):
    stats_rows = []
    if group_vals is None:
        group_vals = sorted(df[group_col].dropna().unique())
    for val in group_vals:
        subset = df[df[group_col] == val]
        n = len(subset)
        if n == 0:
            continue
        diff = subset['diff']
        abs_diff = diff.abs()
        row = {
            group_col: val,
            'n_pixels': n,
            'mae': float(abs_diff.mean()),
            'mse': float((diff ** 2).mean()),
            'rmse': float(np.sqrt((diff ** 2).mean())),
            'bias': float(diff.mean()),
            'std': float(diff.std()),
            'median_ae': float(abs_diff.median()),
            'p90_ae': float(abs_diff.quantile(0.9)),
            'p95_ae': float(abs_diff.quantile(0.95)),
            'p99_ae': float(abs_diff.quantile(0.99)),
            'est_mean': float(subset['est'].mean()),
            'obs_mean': float(subset['obs'].mean()),
        }
        if 'era5_skt' in subset.columns:
            row['era5_mean'] = float(subset['era5_skt'].mean())
        if n > 2 and subset['est'].std() > 0 and subset['obs'].std() > 0:
            row['pearson_r'] = float(subset['est'].corr(subset['obs']))
        else:
            row['pearson_r'] = np.nan
        stats_rows.append(row)
    return pd.DataFrame(stats_rows)


def main():
    parser = argparse.ArgumentParser(
        description='Per-pixel evaluation from test .pt + model checkpoint')
    parser.add_argument('--test-pt', required=True)
    parser.add_argument('--model-folder', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-id', default='unknown')
    parser.add_argument('--era5-bins', type=int, default=30)
    parser.add_argument('--elevation-bins', type=int, default=15)
    parser.add_argument('--urban-threshold', type=float, default=0.5)
    parser.add_argument('--suburban-threshold', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--no-pixel-csv', action='store_true', default=False)
    parser.add_argument('--pixel-subsample', type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load test data ──────────────────────────────────────────────────
    print(f"Loading test data from {args.test_pt}...")
    data = torch.load(args.test_pt, map_location='cpu')

    is_conditioned = 'spatial_inputs' in data and 'conditioning' in data
    norm_params = data['normalisation_parameters']
    normalised = data.get('normalized', True)
    all_input_vars = data['input_variables']
    N = data['n_samples']

    if is_conditioned:
        spatial = data['spatial_inputs']     # (N, C_s, 100, 100)
        cond = data['conditioning']          # (N, C_c)
        spatial_vars = data['spatial_variables']
        cond_vars = data['cond_variables']
        targets = data['outputs']
        H, W = spatial.shape[2], spatial.shape[3]
        # Also build flat inputs for non-conditioned models (LinearModel, standard UNet)
        cond_broadcast = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        inputs = torch.cat([spatial, cond_broadcast], dim=1)
        print(f"  Conditioned format: {N} boxes, {len(spatial_vars)} spatial, "
              f"{len(cond_vars)} cond -> {inputs.shape[1]} flat channels")
    else:
        inputs = data['inputs']              # (N, C, 100, 100)
        targets = data['outputs']
        H, W = inputs.shape[2], inputs.shape[3]
        spatial_vars = all_input_vars
        cond_vars = []
        print(f"  Legacy format: {N} boxes, {len(all_input_vars)} input channels")

    print(f"  Variables: {all_input_vars}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model_folder}...")

    params_path = os.path.join(args.model_folder, 'parameters.json')
    with open(params_path) as f:
        params = json.load(f)
    model_type = params.get('type', 'UNET')

    if model_type == 'LinearModel':
        from cae_tools.models.linear_model import LinearModel
        mt = LinearModel()
        mt.load(args.model_folder)
    else:
        from cae_tools.models.unet import UNET
        mt = UNET()
        mt.load(args.model_folder)

    architecture = getattr(mt, 'architecture', 'standard')
    print(f"  Architecture: {architecture}")

    device = torch.device('cpu')
    if model_type == 'LinearModel':
        mt.weights.to(device).eval()
    else:
        if hasattr(mt, 'encoder') and mt.encoder is not None:
            mt.encoder.to(device).eval()
        if hasattr(mt, 'decoder') and mt.decoder is not None:
            mt.decoder.to(device).eval()

    # ── Run inference ───────────────────────────────────────────────────
    print(f"\nRunning inference on {N} boxes (batch_size={args.batch_size})...")
    predictions = np.zeros((N, 1, H, W), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)

            if architecture == 'conditioned':
                from cae_tools.models.conditioned_helpers import forward_pass
                sp_batch = spatial[start:end].to(device)
                cd_batch = cond[start:end].to(device)
                pred = forward_pass(mt.encoder, mt.decoder, sp_batch, cd_batch)
            elif model_type == 'LinearModel':
                in_batch = inputs[start:end].to(device)
                pred = mt.weights(in_batch)
            else:
                # Standard UNet: encoder -> decoder
                in_batch = inputs[start:end].to(device)
                enc_out, skips = mt.encoder(in_batch)
                pred = mt.decoder(enc_out, skips)

            predictions[start:end] = pred.cpu().numpy()

            if (start // args.batch_size) % 10 == 0:
                print(f"  {end}/{N}...", flush=True)

    print("  Inference done")

    # ── Denormalise ─────────────────────────────────────────────────────
    print("\nDenormalising...")
    pred_phys = denorm_output(predictions, norm_params)
    target_phys = denorm_output(targets.numpy(), norm_params)

    # Denormalise all input variables to physical units
    # Build a unified dict: var_name -> (N, H, W) or (N,) physical values
    var_phys = {}

    if is_conditioned:
        spatial_np = spatial.numpy()
        for i, v in enumerate(spatial_vars):
            var_phys[v] = denorm_value(spatial_np[:, i], v, norm_params)
            # shape (N, 100, 100)

        cond_np = cond.numpy()
        for i, v in enumerate(cond_vars):
            var_phys[v] = denorm_value(cond_np[:, i], v, norm_params)
            # shape (N,) — scalar per box
    else:
        inputs_np = inputs.numpy()
        for i, v in enumerate(all_input_vars):
            var_phys[v] = denorm_value(inputs_np[:, i], v, norm_params)
            # shape (N, 100, 100) — all broadcast in legacy

    # ── Recover DOY and month ───────────────────────────────────────────
    sin_key = 'sin_doy'
    cos_key = 'cos_doy'
    if sin_key in var_phys and cos_key in var_phys:
        sin_vals = var_phys[sin_key]
        cos_vals = var_phys[cos_key]
        # Extract per-box scalar: take pixel [0,0] if spatial, or use directly if scalar
        if sin_vals.ndim == 3:
            sin_box = sin_vals[:, 0, 0]
            cos_box = cos_vals[:, 0, 0]
        else:
            sin_box = sin_vals
            cos_box = cos_vals
        doy, months = recover_doy_and_month(sin_box, cos_box)
    else:
        print("WARNING: sin_doy/cos_doy not found, month recovery unavailable")
        doy = np.zeros(N, dtype=int)
        months = np.zeros(N, dtype=int)

    season_map = {12: 'DJF', 1: 'DJF', 2: 'DJF',
                  3: 'MAM', 4: 'MAM', 5: 'MAM',
                  6: 'JJA', 7: 'JJA', 8: 'JJA',
                  9: 'SON', 10: 'SON', 11: 'SON'}

    # ── Flatten to per-pixel ────────────────────────────────────────────
    print("\nAssembling per-pixel data...")
    t0 = time.time()
    n_pixels = N * H * W

    # Box-level attributes broadcast to all pixels
    box_idx_flat = np.repeat(np.arange(N), H * W)
    month_flat = np.repeat(months, H * W)
    doy_flat = np.repeat(doy, H * W)

    # Predictions and targets
    est_flat = pred_phys[:, 0].ravel()
    obs_flat = target_phys[:, 0].ravel()
    diff_flat = est_flat - obs_flat

    # Flatten each variable
    var_flat = {}
    for v, arr in var_phys.items():
        if arr.ndim == 3:
            # spatial: (N, H, W) -> (N*H*W,)
            var_flat[v] = arr.ravel()
        elif arr.ndim == 1:
            # conditioning scalar: (N,) -> repeat to (N*H*W,)
            var_flat[v] = np.repeat(arr, H * W)
        else:
            print(f"  WARNING: unexpected shape for {v}: {arr.shape}, skipping")

    print(f"  {n_pixels:,} total pixels, took {time.time()-t0:.1f}s")

    # ── Build stats DataFrame ───────────────────────────────────────────
    print("\nBuilding stats DataFrame...")
    stats_df = pd.DataFrame({
        'est': est_flat,
        'obs': obs_flat,
        'diff': diff_flat,
        'abs_diff': np.abs(diff_flat),
        'month': month_flat,
        'doy': doy_flat,
    })

    # Add ERA5 skt
    if 'era5_skt' in var_flat:
        stats_df['era5_skt'] = var_flat['era5_skt']
        stats_df['era5_minus_est'] = var_flat['era5_skt'] - est_flat

    # Land cover
    if 'land_cover' in var_flat:
        stats_df['land_cover'] = np.round(var_flat['land_cover']).astype(np.int64)

    # Urban / suburban
    if 'urban_area' in var_flat:
        stats_df['urban_area'] = var_flat['urban_area']
        stats_df['is_urban'] = (var_flat['urban_area'] >= args.urban_threshold).astype(int)
    if 'suburban_area' in var_flat:
        stats_df['suburban_area'] = var_flat['suburban_area']
        stats_df['is_suburban'] = (var_flat['suburban_area'] >= args.suburban_threshold).astype(int)
    if 'urban_area' in var_flat and 'suburban_area' in var_flat:
        stats_df['is_built'] = ((var_flat['urban_area'] >= args.urban_threshold) |
                                (var_flat['suburban_area'] >= args.suburban_threshold)).astype(int)

    # Elevation
    if 'elevation' in var_flat:
        stats_df['elevation'] = var_flat['elevation']

    stats_df['season'] = stats_df['month'].map(season_map)

    print(f"  Overall MAE:  {stats_df['abs_diff'].mean():.3f} K")
    print(f"  Overall RMSE: {np.sqrt((stats_df['diff']**2).mean()):.3f} K")
    print(f"  Overall bias: {stats_df['diff'].mean():.3f} K")

    # Exclude non-UK pixels (land_cover == 0: France, Ireland, Channel Islands)
    if 'land_cover' in stats_df.columns:
        n_before = len(stats_df)
        stats_df = stats_df[stats_df['land_cover'] != 0].copy()
        print(f"  Excluded {n_before - len(stats_df):,} non-UK pixels (land_cover=0)")
        print(f"  UK-only: {len(stats_df):,} pixels")
        print(f"  UK MAE:  {stats_df['abs_diff'].mean():.3f} K")
        print(f"  UK RMSE: {np.sqrt((stats_df['diff']**2).mean()):.3f} K")
        print(f"  UK bias: {stats_df['diff'].mean():.3f} K")

    # ── Per-pixel CSV ───────────────────────────────────────────────────
    if not args.no_pixel_csv:
        print("\nBuilding per-pixel CSV...")
        pixel_data = {
            'box_idx': box_idx_flat,
            'month': month_flat,
            'doy': doy_flat,
            'est': est_flat,
            'obs': obs_flat,
            'diff': diff_flat,
        }
        for v in all_input_vars:
            if v in var_flat:
                pixel_data[v] = var_flat[v]

        if args.pixel_subsample:
            print(f"  Subsampling {args.pixel_subsample} pixels per box...")
            rng = np.random.RandomState(42)
            keep = []
            for b in range(N):
                base = b * H * W
                idx = rng.choice(H * W, size=min(args.pixel_subsample, H * W),
                                 replace=False) + base
                keep.append(idx)
            keep = np.concatenate(keep)
            pixel_df = pd.DataFrame({k: v[keep] for k, v in pixel_data.items()})
        else:
            pixel_df = pd.DataFrame(pixel_data)

        try:
            pixel_path = os.path.join(args.output_dir, 'eval_pixels.parquet')
            print(f"  Saving to {pixel_path}...")
            pixel_df.to_parquet(pixel_path, index=False, engine='pyarrow')
            size_mb = os.path.getsize(pixel_path) / 1e6
            print(f"  {size_mb:.1f} MB ({len(pixel_df):,} pixels)")
        except Exception as e:
            print(f"  WARNING: Failed to save parquet: {e}")
            print(f"  Continuing with stats only...")

    # ── Aggregated statistics ───────────────────────────────────────────
    print("\nComputing aggregated statistics...")

    # By month
    s = compute_stats(stats_df, 'month', list(range(1, 13)))
    path = os.path.join(args.output_dir, 'eval_stats_by_month.csv')
    s.to_csv(path, index=False, float_format='%.4f')
    print(f"\n  By month:")
    print(s[['month', 'n_pixels', 'mae', 'rmse', 'bias']].to_string(index=False))

    # By season
    s = compute_stats(stats_df, 'season', ['DJF', 'MAM', 'JJA', 'SON'])
    path = os.path.join(args.output_dir, 'eval_stats_by_season.csv')
    s.to_csv(path, index=False, float_format='%.4f')
    print(f"\n  By season:")
    print(s[['season', 'n_pixels', 'mae', 'rmse', 'bias']].to_string(index=False))

    # By land class
    if 'land_cover' in stats_df.columns:
        lc_names = {
            1: 'Deciduous_woodland', 2: 'Coniferous_woodland',
            3: 'Arable', 4: 'Improved_grassland', 5: 'Neutral_grassland',
            6: 'Calcareous_grassland', 7: 'Acid_grassland', 8: 'Fen',
            9: 'Heather', 10: 'Heather_grassland', 11: 'Bog',
            12: 'Inland_rock', 13: 'Saltwater', 14: 'Freshwater',
            15: 'Supralittoral_rock', 16: 'Supralittoral_sediment',
            17: 'Littoral_rock', 18: 'Littoral_sediment',
            19: 'Saltmarsh', 20: 'Urban', 21: 'Suburban', 22: 'Missing'
        }
        s = compute_stats(stats_df, 'land_cover')
        s['land_cover_name'] = s['land_cover'].map(lc_names)
        path = os.path.join(args.output_dir, 'eval_stats_by_landclass.csv')
        s.to_csv(path, index=False, float_format='%.4f')
        print(f"\n  By land class:")
        print(s[['land_cover', 'land_cover_name', 'n_pixels', 'mae', 'rmse', 'bias']]
              .to_string(index=False))

    # Urban vs rural by month
    if 'is_urban' in stats_df.columns:
        urban_df = stats_df[stats_df['is_urban'] == 1]
        rural_df = stats_df[stats_df['is_urban'] == 0]
        u_monthly = compute_stats(urban_df, 'month', list(range(1, 13)))
        u_monthly['type'] = 'urban'
        r_monthly = compute_stats(rural_df, 'month', list(range(1, 13)))
        r_monthly['type'] = 'rural'
        combined = pd.concat([u_monthly, r_monthly])
        path = os.path.join(args.output_dir, 'eval_stats_urban_by_month.csv')
        combined.to_csv(path, index=False, float_format='%.4f')
        print(f"\n  Urban vs rural by month:")
        print(combined[['type', 'month', 'n_pixels', 'mae', 'rmse', 'bias']]
              .to_string(index=False))

    # Built overall
    if 'is_built' in stats_df.columns:
        s = compute_stats(stats_df, 'is_built')
        s['label'] = s['is_built'].map({0: 'rural', 1: 'built'})
        path = os.path.join(args.output_dir, 'eval_stats_by_built.csv')
        s.to_csv(path, index=False, float_format='%.4f')
        print(f"\n  By built/rural:")
        print(s[['label', 'n_pixels', 'mae', 'rmse', 'bias']].to_string(index=False))

    # Urban by season
    if 'is_urban' in stats_df.columns:
        rows = []
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            for utype, ulabel in [(0, 'rural'), (1, 'urban')]:
                subset = stats_df[(stats_df['season'] == season) &
                                  (stats_df['is_urban'] == utype)]
                if len(subset) == 0:
                    continue
                diff = subset['diff']
                rows.append({
                    'season': season, 'type': ulabel,
                    'n_pixels': len(subset),
                    'mae': float(diff.abs().mean()),
                    'rmse': float(np.sqrt((diff ** 2).mean())),
                    'bias': float(diff.mean()),
                })
        cross = pd.DataFrame(rows)
        path = os.path.join(args.output_dir, 'eval_stats_urban_by_season.csv')
        cross.to_csv(path, index=False, float_format='%.4f')
        print(f"\n  Urban by season:")
        print(cross.to_string(index=False))

    # By ERA5 - est bin
    if 'era5_minus_est' in stats_df.columns:
        valid = stats_df['era5_minus_est'].notna()
        if valid.sum() > 0:
            stats_df.loc[valid, 'era5_est_bin'] = pd.cut(
                stats_df.loc[valid, 'era5_minus_est'],
                bins=args.era5_bins, labels=False)
            bin_edges = pd.cut(stats_df.loc[valid, 'era5_minus_est'],
                               bins=args.era5_bins).cat.categories
            s = compute_stats(stats_df[valid], 'era5_est_bin')
            s['bin_left'] = [bin_edges[int(i)].left for i in s['era5_est_bin']]
            s['bin_right'] = [bin_edges[int(i)].right for i in s['era5_est_bin']]
            path = os.path.join(args.output_dir, 'eval_stats_by_era5_est_bin.csv')
            s.to_csv(path, index=False, float_format='%.4f')
            print(f"\n  By ERA5-est bin -> {path}")

    # By elevation bin
    if 'elevation' in stats_df.columns:
        valid = stats_df['elevation'].notna()
        if valid.sum() > 0:
            stats_df.loc[valid, 'elev_bin'] = pd.cut(
                stats_df.loc[valid, 'elevation'],
                bins=args.elevation_bins, labels=False)
            bin_edges = pd.cut(stats_df.loc[valid, 'elevation'],
                               bins=args.elevation_bins).cat.categories
            s = compute_stats(stats_df[valid], 'elev_bin')
            s['bin_left'] = [bin_edges[int(i)].left for i in s['elev_bin']]
            s['bin_right'] = [bin_edges[int(i)].right for i in s['elev_bin']]
            path = os.path.join(args.output_dir, 'eval_stats_by_elevation_bin.csv')
            s.to_csv(path, index=False, float_format='%.4f')
            print(f"\n  By elevation bin -> {path}")

    # ── Summary JSON ────────────────────────────────────────────────────
    summary = {
        'model_id': args.model_id,
        'model_folder': args.model_folder,
        'test_pt': args.test_pt,
        'architecture': architecture,
        'format': 'conditioned' if is_conditioned else 'legacy',
        'n_boxes': N,
        'n_pixels': n_pixels,
        'input_variables': all_input_vars,
        'overall_mae': float(stats_df['abs_diff'].mean()),
        'overall_rmse': float(np.sqrt((stats_df['diff'] ** 2).mean())),
        'overall_bias': float(stats_df['diff'].mean()),
        'overall_std': float(stats_df['diff'].std()),
        'overall_median_ae': float(stats_df['abs_diff'].median()),
        'overall_p90_ae': float(stats_df['abs_diff'].quantile(0.9)),
        'urban_threshold': args.urban_threshold,
        'suburban_threshold': args.suburban_threshold,
    }
    if stats_df['est'].std() > 0 and stats_df['obs'].std() > 0:
        summary['overall_pearson_r'] = float(stats_df['est'].corr(stats_df['obs']))

    path = os.path.join(args.output_dir, 'eval_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary -> {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
