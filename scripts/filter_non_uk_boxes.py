"""
Filter non-UK boxes from preprocessed .pt files.

Removes boxes where >threshold fraction of pixels have land_cover == 0
(France, Ireland, Channel Islands — outside UKCEH classification boundary).
Recalculates normalisation parameters from remaining UK-only data.

Supports both legacy (v8: 'inputs') and conditioned (v9: 'spatial_inputs' +
'conditioning') formats.

Usage:
  # Single file
  python filter_non_uk_boxes.py \
      --input-pt /path/to/train.pt \
      --output-pt /path/to/train_uk.pt

  # Batch mode: process multiple files
  python filter_non_uk_boxes.py \
      --input-pt file1.pt file2.pt file3.pt \
      --suffix _uk \
      --threshold 0.5

  # Use normalisation from a reference file (e.g. apply train norms to test/val)
  python filter_non_uk_boxes.py \
      --input-pt test.pt \
      --output-pt test_uk.pt \
      --use-norm-from train_uk.pt
"""

import argparse
import os
import sys
import time
import numpy as np
import torch


def detect_format(data):
    """Detect .pt format: 'conditioned' or 'legacy'."""
    if 'spatial_inputs' in data and 'conditioning' in data:
        return 'conditioned'
    elif 'inputs' in data:
        return 'legacy'
    else:
        raise ValueError(f"Unknown .pt format. Keys: {list(data.keys())}")


def get_land_cover(data, fmt, norm_params):
    """Extract and denormalise land_cover channel. Returns (N, H, W) int array."""
    if fmt == 'conditioned':
        sp_vars = data['spatial_variables']
        lc_idx = sp_vars.index('land_cover')
        lc_norm = data['spatial_inputs'][:, lc_idx].numpy()
    else:
        iv = data['input_variables']
        lc_idx = iv.index('land_cover')
        lc_norm = data['inputs'][:, lc_idx].numpy()

    mn = norm_params['min_inputs']['land_cover']
    mx = norm_params['max_inputs']['land_cover']
    lc_phys = lc_norm * (mx - mn) + mn
    return np.round(lc_phys).astype(int)


def compute_normalisation(data, fmt, keep_mask):
    """Recompute normalisation parameters from kept boxes only.

    Returns new norm_params dict with updated min/max for all variables.
    Operates on raw (denormalised) data.
    """
    old_norm = data['normalisation_parameters']
    new_norm = {}

    # Copy non-min/max keys
    for k, v in old_norm.items():
        if k not in ('min_inputs', 'max_inputs', 'min_output', 'max_output'):
            new_norm[k] = v

    # ── Denormalise inputs and recompute min/max ──
    if fmt == 'conditioned':
        spatial = data['spatial_inputs'][keep_mask]
        cond = data['conditioning'][keep_mask]
        sp_vars = data['spatial_variables']
        cond_vars = data['cond_variables']

        new_min_inputs = {}
        new_max_inputs = {}

        # Spatial variables
        for i, v in enumerate(sp_vars):
            mn = old_norm['min_inputs'][v]
            mx = old_norm['max_inputs'][v]
            phys = spatial[:, i].numpy() * (mx - mn) + mn
            new_min_inputs[v] = float(np.nanmin(phys))
            new_max_inputs[v] = float(np.nanmax(phys))

        # Conditioning variables
        for i, v in enumerate(cond_vars):
            mn = old_norm['min_inputs'][v]
            mx = old_norm['max_inputs'][v]
            phys = cond[:, i].numpy() * (mx - mn) + mn
            new_min_inputs[v] = float(np.nanmin(phys))
            new_max_inputs[v] = float(np.nanmax(phys))

    else:
        inputs = data['inputs'][keep_mask]
        iv = data['input_variables']

        new_min_inputs = {}
        new_max_inputs = {}

        for i, v in enumerate(iv):
            mn = old_norm['min_inputs'][v]
            mx = old_norm['max_inputs'][v]
            phys = inputs[:, i].numpy() * (mx - mn) + mn
            new_min_inputs[v] = float(np.nanmin(phys))
            new_max_inputs[v] = float(np.nanmax(phys))

    # Output
    outputs = data['outputs'][keep_mask]
    mn_out = old_norm['min_output']
    mx_out = old_norm['max_output']
    activation = old_norm.get('output_activation', 'sigmoid')
    if activation == 'tanh':
        phys_out = mn_out + ((outputs.numpy() + 1) / 2) * (mx_out - mn_out)
    else:
        phys_out = outputs.numpy() * (mx_out - mn_out) + mn_out
    new_min_output = float(np.nanmin(phys_out))
    new_max_output = float(np.nanmax(phys_out))

    new_norm['min_inputs'] = new_min_inputs
    new_norm['max_inputs'] = new_max_inputs
    new_norm['min_output'] = new_min_output
    new_norm['max_output'] = new_max_output

    return new_norm


def renormalise_data(data, fmt, old_norm, new_norm):
    """Denormalise with old params, renormalise with new params. In-place."""
    if fmt == 'conditioned':
        spatial = data['spatial_inputs']
        cond = data['conditioning']
        sp_vars = data['spatial_variables']
        cond_vars = data['cond_variables']

        for i, v in enumerate(sp_vars):
            old_mn = old_norm['min_inputs'][v]
            old_mx = old_norm['max_inputs'][v]
            new_mn = new_norm['min_inputs'][v]
            new_mx = new_norm['max_inputs'][v]
            # Denormalise
            phys = spatial[:, i] * (old_mx - old_mn) + old_mn
            # Renormalise
            if new_mx - new_mn > 0:
                spatial[:, i] = (phys - new_mn) / (new_mx - new_mn)
            else:
                spatial[:, i] = 0.0

        for i, v in enumerate(cond_vars):
            old_mn = old_norm['min_inputs'][v]
            old_mx = old_norm['max_inputs'][v]
            new_mn = new_norm['min_inputs'][v]
            new_mx = new_norm['max_inputs'][v]
            phys = cond[:, i] * (old_mx - old_mn) + old_mn
            if new_mx - new_mn > 0:
                cond[:, i] = (phys - new_mn) / (new_mx - new_mn)
            else:
                cond[:, i] = 0.0

    else:
        inputs = data['inputs']
        iv = data['input_variables']

        for i, v in enumerate(iv):
            old_mn = old_norm['min_inputs'][v]
            old_mx = old_norm['max_inputs'][v]
            new_mn = new_norm['min_inputs'][v]
            new_mx = new_norm['max_inputs'][v]
            phys = inputs[:, i] * (old_mx - old_mn) + old_mn
            if new_mx - new_mn > 0:
                inputs[:, i] = (phys - new_mn) / (new_mx - new_mn)
            else:
                inputs[:, i] = 0.0

    # Output
    outputs = data['outputs']
    old_mn_out = old_norm['min_output']
    old_mx_out = old_norm['max_output']
    new_mn_out = new_norm['min_output']
    new_mx_out = new_norm['max_output']
    activation = old_norm.get('output_activation', 'sigmoid')

    if activation == 'tanh':
        phys_out = old_mn_out + ((outputs + 1) / 2) * (old_mx_out - old_mn_out)
        outputs[:] = 2 * (phys_out - new_mn_out) / (new_mx_out - new_mn_out) - 1
    else:
        phys_out = outputs * (old_mx_out - old_mn_out) + old_mn_out
        if new_mx_out - new_mn_out > 0:
            outputs[:] = (phys_out - new_mn_out) / (new_mx_out - new_mn_out)
        else:
            outputs[:] = 0.0


def process_file(input_path, output_path, threshold, ref_norm=None):
    """Process a single .pt file: filter non-UK boxes, renormalise, save."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"Output:     {output_path}")

    t0 = time.time()
    data = torch.load(input_path, map_location='cpu')
    fmt = detect_format(data)
    print(f"  Format: {fmt}")

    old_norm = data['normalisation_parameters']

    # ── Identify non-UK boxes ──
    lc_int = get_land_cover(data, fmt, old_norm)  # (N, H, W)
    N = lc_int.shape[0]
    frac_zero = (lc_int == 0).reshape(N, -1).mean(axis=1)

    keep_mask = frac_zero <= threshold
    n_keep = int(keep_mask.sum())
    n_remove = N - n_keep

    print(f"  Total boxes:   {N:,}")
    print(f"  Non-UK (>{threshold*100:.0f}% lc=0): {n_remove:,} ({100*n_remove/N:.1f}%)")
    print(f"  Keeping:       {n_keep:,} ({100*n_keep/N:.1f}%)")

    if n_remove == 0:
        print("  No non-UK boxes found, skipping.")
        return

    # ── Compute new normalisation from UK-only data ──
    if ref_norm is not None:
        new_norm = ref_norm
        print("  Using normalisation from reference file")
    else:
        print("  Recomputing normalisation from UK-only data...")
        new_norm = compute_normalisation(data, fmt, keep_mask)

    # Print normalisation changes
    print("  Normalisation changes:")
    for v in sorted(old_norm['min_inputs'].keys()):
        old_mn = old_norm['min_inputs'][v]
        old_mx = old_norm['max_inputs'][v]
        new_mn = new_norm['min_inputs'][v]
        new_mx = new_norm['max_inputs'][v]
        if abs(old_mn - new_mn) > 1e-6 or abs(old_mx - new_mx) > 1e-6:
            print(f"    {v}: [{old_mn:.4f}, {old_mx:.4f}] -> [{new_mn:.4f}, {new_mx:.4f}]")
    old_mn_o = old_norm['min_output']
    old_mx_o = old_norm['max_output']
    new_mn_o = new_norm['min_output']
    new_mx_o = new_norm['max_output']
    if abs(old_mn_o - new_mn_o) > 1e-6 or abs(old_mx_o - new_mx_o) > 1e-6:
        print(f"    output: [{old_mn_o:.4f}, {old_mx_o:.4f}] -> [{new_mn_o:.4f}, {new_mx_o:.4f}]")

    # ── Filter boxes ──
    keep_idx = torch.where(torch.tensor(keep_mask))[0]

    if fmt == 'conditioned':
        data['spatial_inputs'] = data['spatial_inputs'][keep_idx]
        data['conditioning'] = data['conditioning'][keep_idx]
    else:
        data['inputs'] = data['inputs'][keep_idx]
    data['outputs'] = data['outputs'][keep_idx]

    # ── Renormalise with new parameters ──
    print("  Renormalising data...")
    renormalise_data(data, fmt, old_norm, new_norm)
    data['normalisation_parameters'] = new_norm
    data['n_samples'] = n_keep

    # ── Preserve other metadata ──
    if 'cloud_filter_meta' in data:
        data['cloud_filter_meta']['n_total_before_uk_filter'] = N
        data['cloud_filter_meta']['n_removed_non_uk'] = n_remove
        data['cloud_filter_meta']['uk_filter_threshold'] = threshold

    # Add filtering metadata
    data['uk_filter_meta'] = {
        'threshold': threshold,
        'n_original': N,
        'n_removed': n_remove,
        'n_kept': n_keep,
        'method': 'land_cover == 0 fraction > threshold',
    }

    # ── Save ──
    print(f"  Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(data, output_path)

    size_gb = os.path.getsize(output_path) / 1e9
    elapsed = time.time() - t0
    print(f"  Done: {n_keep:,} boxes, {size_gb:.2f} GB, {elapsed:.0f}s")

    return new_norm


def main():
    parser = argparse.ArgumentParser(
        description='Filter non-UK boxes from preprocessed .pt files')
    parser.add_argument('--input-pt', nargs='+', required=True,
                        help='One or more input .pt files')
    parser.add_argument('--output-pt', nargs='*', default=None,
                        help='Output paths (one per input). If not specified, '
                             'uses --suffix to generate output names.')
    parser.add_argument('--suffix', default='_uk',
                        help='Suffix for output files when --output-pt not given '
                             '(default: _uk)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Remove boxes with more than this fraction of '
                             'land_cover=0 pixels (default: 0.5)')
    parser.add_argument('--use-norm-from', default=None,
                        help='Path to a .pt file whose normalisation parameters '
                             'should be used for all output files (e.g. use '
                             'train norms for test/val)')
    parser.add_argument('--share-train-norm', action='store_true', default=False,
                        help='Process first file as train (compute norms), '
                             'then apply those norms to all subsequent files')

    args = parser.parse_args()

    # Build output paths
    if args.output_pt:
        if len(args.output_pt) != len(args.input_pt):
            print(f"ERROR: {len(args.input_pt)} inputs but {len(args.output_pt)} outputs")
            sys.exit(1)
        output_paths = args.output_pt
    else:
        output_paths = []
        for p in args.input_pt:
            base, ext = os.path.splitext(p)
            output_paths.append(f"{base}{args.suffix}{ext}")

    # Load reference normalisation if specified
    ref_norm = None
    if args.use_norm_from:
        print(f"Loading reference normalisation from {args.use_norm_from}...")
        ref_data = torch.load(args.use_norm_from, map_location='cpu')
        ref_norm = ref_data['normalisation_parameters']
        del ref_data

    # Process files
    train_norm = None
    for i, (inp, outp) in enumerate(zip(args.input_pt, output_paths)):
        if args.share_train_norm and i == 0:
            # First file: compute norms (treat as train)
            train_norm = process_file(inp, outp, args.threshold, ref_norm=ref_norm)
        elif args.share_train_norm and i > 0:
            # Subsequent files: use train norms
            process_file(inp, outp, args.threshold, ref_norm=train_norm)
        else:
            process_file(inp, outp, args.threshold, ref_norm=ref_norm)

    print(f"\n{'='*60}")
    print("All files processed.")


if __name__ == '__main__':
    main()
