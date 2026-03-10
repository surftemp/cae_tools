"""
Create a balanced subset of the training/test/val .pt files for local development.

Stratified sampling to ensure good representation across:
  - Month (temporal balance)
  - Latitude band (spatial balance)  
  - Dominant land cover class (class balance)

Usage:
    python create_local_dataset.py \
        --input-pt /path/to/train.pt \
        --output-pt /path/to/train_local.pt \
        --fraction 0.25

    # Or specify exact count
    python create_local_dataset.py \
        --input-pt /path/to/train.pt \
        --output-pt /path/to/train_local.pt \
        --n-boxes 10000
"""

import argparse
import os
import sys
import numpy as np
import torch


def recover_month(sin_vals, cos_vals):
    """Recover month from sin/cos DOY."""
    d1 = np.arcsin(np.clip(sin_vals, 0, 1)) * 366 / np.pi
    d2 = 366 - d1
    c1 = np.cos((d1 / 366) * 2 * np.pi)
    c2 = np.cos((d2 / 366) * 2 * np.pi)
    doy = np.where(np.abs(c1 - cos_vals) < np.abs(c2 - cos_vals), d1, d2)
    doy = np.clip(doy, 1, 366).astype(int)

    boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]
    months = np.zeros(len(doy), dtype=int)
    for m in range(12):
        mask = (doy > boundaries[m]) & (doy <= boundaries[m + 1])
        months[mask] = m + 1
    months[months == 0] = 1
    return months


def get_lat_band(elevation_map):
    """
    Rough proxy for latitude using mean elevation pattern.
    Can't recover actual lat from the .pt file, so use the box's
    mean output temperature as a latitude proxy (colder = more north).
    Returns bin index 0-4.
    """
    # This is a placeholder — actual lat isn't in the .pt
    # We'll use output mean temperature as proxy
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced subset for local development')
    parser.add_argument('--input-pt', required=True, nargs='+',
                        help='Input .pt file(s) (train, test, val)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for subset files')
    parser.add_argument('--fraction', type=float, default=0.25,
                        help='Fraction of boxes to keep (default: 0.25)')
    parser.add_argument('--n-boxes', type=int, default=None,
                        help='Exact number of boxes (overrides --fraction)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    for pt_path in args.input_pt:
        basename = os.path.basename(pt_path)
        out_path = os.path.join(args.output_dir, basename.replace('.pt', '_local.pt'))

        print(f"\n{'='*60}")
        print(f"Processing: {basename}")
        print(f"{'='*60}")

        data = torch.load(pt_path, map_location='cpu')
        N = data['n_samples']
        norm = data['normalisation_parameters']
        is_conditioned = 'spatial_inputs' in data

        # ── Determine target count ──
        if args.n_boxes:
            n_target = min(args.n_boxes, N)
        else:
            n_target = int(N * args.fraction)
        print(f"  Total: {N:,} boxes, target: {n_target:,} ({100*n_target/N:.0f}%)")

        # ── Extract stratification variables ──

        # 1. Month from sin/cos DOY
        if is_conditioned:
            cond = data['conditioning'].numpy()
            cond_vars = data['cond_variables']
            sin_idx = cond_vars.index('sin_doy')
            cos_idx = cond_vars.index('cos_doy')
            sin_phys = cond[:, sin_idx] * (norm['max_inputs']['sin_doy'] - norm['min_inputs']['sin_doy']) + norm['min_inputs']['sin_doy']
            cos_phys = cond[:, cos_idx] * (norm['max_inputs']['cos_doy'] - norm['min_inputs']['cos_doy']) + norm['min_inputs']['cos_doy']
        else:
            inputs = data['inputs'].numpy()
            iv = data['input_variables']
            sin_idx = iv.index('sin_doy')
            cos_idx = iv.index('cos_doy')
            sin_phys = inputs[:, sin_idx, 0, 0] * (norm['max_inputs']['sin_doy'] - norm['min_inputs']['sin_doy']) + norm['min_inputs']['sin_doy']
            cos_phys = inputs[:, cos_idx, 0, 0] * (norm['max_inputs']['cos_doy'] - norm['min_inputs']['cos_doy']) + norm['min_inputs']['cos_doy']

        months = recover_month(sin_phys, cos_phys)

        # 2. Dominant land cover class per box
        if is_conditioned:
            sp_vars = data['spatial_variables']
            lc_idx = sp_vars.index('land_cover')
            lc_norm = data['spatial_inputs'][:, lc_idx].numpy()
        else:
            iv = data['input_variables']
            lc_idx = iv.index('land_cover')
            lc_norm = data['inputs'][:, lc_idx].numpy()

        lc_mn = norm['min_inputs']['land_cover']
        lc_mx = norm['max_inputs']['land_cover']
        lc_phys = lc_norm * (lc_mx - lc_mn) + lc_mn
        # Dominant class = mode of rounded values per box
        lc_flat = np.round(lc_phys.reshape(N, -1)).astype(int)
        dominant_lc = np.array([np.bincount(row[row > 0], minlength=23).argmax()
                                if (row > 0).any() else 0
                                for row in lc_flat])

        # Bin land cover into groups for stratification
        # 1-2: woodland, 3-4: arable/grassland, 5-11: semi-natural,
        # 12-19: rock/water/coastal, 20-21: urban/suburban
        def lc_group(lc):
            if lc <= 2:
                return 'woodland'
            elif lc <= 4:
                return 'arable_grass'
            elif lc <= 11:
                return 'semi_natural'
            elif lc <= 19:
                return 'rock_water_coast'
            elif lc <= 21:
                return 'urban_suburban'
            else:
                return 'other'

        lc_groups = np.array([lc_group(lc) for lc in dominant_lc])

        # 3. Temperature band as latitude proxy
        outputs = data['outputs'].numpy()
        out_mn = norm['min_output']
        out_mx = norm['max_output']
        box_mean_temp = np.nanmean(outputs.reshape(N, -1), axis=1) * (out_mx - out_mn) + out_mn
        temp_bins = np.digitize(box_mean_temp,
                                np.percentile(box_mean_temp, [20, 40, 60, 80]))

        # ── Build strata keys ──
        strata = [f"{m}_{lc}_{t}" for m, lc, t in
                  zip(months, lc_groups, temp_bins)]
        unique_strata = sorted(set(strata))
        strata_arr = np.array(strata)

        print(f"  Unique strata: {len(unique_strata)}")
        print(f"  Month distribution: {dict(zip(*np.unique(months, return_counts=True)))}")
        print(f"  LC group distribution: {dict(zip(*np.unique(lc_groups, return_counts=True)))}")

        # ── Stratified sampling ──
        # Per stratum: sample proportionally, but ensure at least 1 per stratum
        selected = []
        per_stratum = max(1, n_target // len(unique_strata))

        for s in unique_strata:
            idx = np.where(strata_arr == s)[0]
            n_take = min(len(idx), max(1, int(len(idx) * n_target / N)))
            chosen = rng.choice(idx, size=n_take, replace=False)
            selected.append(chosen)

        selected = np.concatenate(selected)

        # If we overshot or undershot, adjust
        if len(selected) > n_target:
            selected = rng.choice(selected, size=n_target, replace=False)
        elif len(selected) < n_target:
            remaining = np.setdiff1d(np.arange(N), selected)
            extra = rng.choice(remaining, size=n_target - len(selected),
                               replace=False)
            selected = np.concatenate([selected, extra])

        selected = np.sort(selected)
        print(f"  Selected: {len(selected):,} boxes")

        # ── Verify balance ──
        sel_months = months[selected]
        sel_lc = lc_groups[selected]
        print(f"\n  Balance check:")
        print(f"  Month distribution (selected):")
        for m, c in sorted(zip(*np.unique(sel_months, return_counts=True))):
            orig = (months == m).sum()
            print(f"    Month {m:2d}: {c:5d} / {orig:5d} "
                  f"({100*c/orig:.0f}% kept, {100*c/len(selected):.1f}% of subset)")

        print(f"  LC group distribution (selected):")
        for lc, c in sorted(zip(*np.unique(sel_lc, return_counts=True))):
            orig = (lc_groups == lc).sum()
            print(f"    {lc:>20s}: {c:5d} / {orig:5d} "
                  f"({100*c/orig:.0f}% kept, {100*c/len(selected):.1f}% of subset)")

        # ── Build output ──
        keep = torch.tensor(selected)
        out_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == N:
                out_data[k] = v[keep]
            else:
                out_data[k] = v
        out_data['n_samples'] = len(selected)
        out_data['local_subset_meta'] = {
            'source': pt_path,
            'fraction': args.fraction,
            'n_original': N,
            'n_selected': len(selected),
            'seed': args.seed,
            'stratification': 'month x lc_group x temp_band',
        }

        print(f"\n  Saving to {out_path}...")
        torch.save(out_data, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {size_mb:.0f} MB")

    print(f"\n{'='*60}")
    print("Done. Download files from:")
    print(f"  {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()