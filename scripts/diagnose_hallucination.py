"""
Diagnostic: Check inputs and channel attention on hallucinating vs normal boxes.

Hypothesis testing:
1. Is there an unseen land cover class (-1, 255, NaN, etc.) in inference data?
2. Are any input channels out of training range on bad dates?
3. Does channel attention suppress ERA5 temperature on bad boxes?
4. What is the actual ERA5 skt value going INTO the model for bad vs good boxes?

Usage:
    # Quick mode: scan all boxes for a date and flag anomalies
    python diagnose_hallucination.py scan \
        --model-folder /path/to/checkpoint \
        --box-dir /path/to/model_inputs/2023-08-04

    # Compare mode: compare a hallucinating box vs a normal box
    python diagnose_hallucination.py compare \
        --model-folder /path/to/checkpoint \
        --bad-box /path/to/bad_box.nc \
        --good-box /path/to/good_box.nc
        
    # Landcover probe: check all unique land cover values across inference dates
    python diagnose_hallucination.py landcover \
        --box-dir /path/to/model_inputs/2023-08-04 \
        --box-dir2 /path/to/model_inputs/2023-08-09
"""

import argparse
import os
import json
import sys
import numpy as np
import torch
import xarray as xr

# Add cae_tools to path if needed
from cae_tools.models.unet import UNET
from cae_tools.models.ds_dataset import DSDataset


def load_model(model_folder):
    """Load model and return it with device."""
    mt = UNET()
    mt.load(model_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt.encoder.to(device)
    mt.decoder.to(device)
    mt.encoder.eval()
    mt.decoder.eval()
    return mt, device


def get_norm_params(mt):
    """Extract min/max dicts from normalisation parameters."""
    if isinstance(mt.normalisation_parameters, dict):
        return (mt.normalisation_parameters['min_inputs'],
                mt.normalisation_parameters['max_inputs'],
                mt.normalisation_parameters['min_output'],
                mt.normalisation_parameters['max_output'])
    else:
        return tuple(mt.normalisation_parameters)


def infer_box(mt, device, score_ds, input_variable_names):
    """Run inference on a single box, return temperature and attention weights."""
    ds = DSDataset(score_ds, input_variable_names, input_variable_names[0],
                   normalise_in=mt.normalise_input)
    ds.set_normalisation_parameters(mt.normalisation_parameters)
    
    in_arr, _, _ = ds[0]
    input_tensor = torch.tensor(in_arr).unsqueeze(0).to(device)
    
    # Hook attention layers
    attention_weights = {}
    hooks = []
    for idx, attn_layer in enumerate(mt.decoder.attention_layers):
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                attention_weights[layer_idx] = out.detach().cpu().numpy()[0, :, 0, 0]
            return hook_fn
        hooks.append(attn_layer.register_forward_hook(make_hook(idx)))
    
    with torch.no_grad():
        encoded, skip = mt.encoder(input_tensor)
        output = mt.decoder(encoded, skip)
    
    for h in hooks:
        h.remove()
    
    # Denormalize
    _, _, min_out, max_out = get_norm_params(mt)
    temp = min_out + output.cpu().numpy() * (max_out - min_out)
    
    return {
        'temp_mean': float(temp.mean()),
        'temp_min': float(temp.min()),
        'temp_max': float(temp.max()),
        'normalized_input': input_tensor.cpu().numpy()[0],
        'attention_weights': attention_weights,
    }


def check_landcover(score_ds):
    """Check for unexpected land cover values."""
    if 'land_cover' not in score_ds:
        return {}
    
    lc = score_ds['land_cover'].values
    unique_vals = np.unique(lc[~np.isnan(lc)])
    
    results = {
        'unique_values': sorted(unique_vals.tolist()),
        'min': float(np.nanmin(lc)),
        'max': float(np.nanmax(lc)),
        'nan_count': int(np.sum(np.isnan(lc))),
        'negative_count': int(np.sum(lc < 0)),
        'above_21_count': int(np.sum(lc > 21)),
        'total_pixels': lc.size,
    }
    return results


def check_input_ranges(score_ds, input_variable_names, min_inputs, max_inputs):
    """Check which inputs are out of training range."""
    out_of_range = {}
    for var_name in input_variable_names:
        if var_name not in score_ds:
            continue
        data = score_ds[var_name].values
        below = int(np.sum(data < min_inputs[var_name]))
        above = int(np.sum(data > max_inputs[var_name]))
        total = data.size
        if below > 0 or above > 0:
            out_of_range[var_name] = {
                'below_pct': 100 * below / total,
                'above_pct': 100 * above / total,
                'data_range': [float(np.nanmin(data)), float(np.nanmax(data))],
                'train_range': [min_inputs[var_name], max_inputs[var_name]],
            }
    return out_of_range


def cmd_scan(args):
    """Scan all boxes in a directory and flag anomalies."""
    mt, device = load_model(args.model_folder)
    input_vars = mt.get_input_variable_names()
    min_inputs, max_inputs, min_out, max_out = get_norm_params(mt)
    
    box_files = sorted([f for f in os.listdir(args.box_dir) if f.endswith('.nc')])
    print(f"Scanning {len(box_files)} boxes in {args.box_dir}")
    print(f"Input variables: {input_vars}")
    print(f"Output range: [{min_out:.2f}K, {max_out:.2f}K]")
    print()
    
    cold_boxes = []
    lc_anomalies = []
    input_anomalies = []
    
    for bf in box_files:
        path = os.path.join(args.box_dir, bf)
        score_ds = xr.open_dataset(path)
        
        # Check land cover
        lc_info = check_landcover(score_ds)
        if lc_info.get('negative_count', 0) > 0 or lc_info.get('above_21_count', 0) > 0 or lc_info.get('nan_count', 0) > 0:
            lc_anomalies.append((bf, lc_info))
        
        # Check input ranges
        oor = check_input_ranges(score_ds, input_vars, min_inputs, max_inputs)
        if oor:
            input_anomalies.append((bf, oor))
        
        # Run inference
        result = infer_box(mt, device, score_ds, input_vars)
        
        if result['temp_mean'] < 285:  # Suspicious for summer
            # Also get raw ERA5 for comparison
            era5_raw = float(score_ds['era5_skt'].values.mean()) if 'era5_skt' in score_ds else None
            cold_boxes.append({
                'file': bf,
                'pred_mean': result['temp_mean'],
                'pred_min': result['temp_min'],
                'era5_raw': era5_raw,
                'delta': result['temp_mean'] - era5_raw if era5_raw else None,
            })
        
        score_ds.close()
    
    # Report
    print(f"{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    
    print(f"\n--- COLD BOXES (predicted mean < 285K) ---")
    if cold_boxes:
        for cb in sorted(cold_boxes, key=lambda x: x['pred_mean']):
            era5_str = f"era5={cb['era5_raw']:.1f}K  delta={cb['delta']:.1f}K" if cb['era5_raw'] else ""
            print(f"  {cb['file']:50s}  pred_mean={cb['pred_mean']:.1f}K  {era5_str}")
        print(f"\n  Total: {len(cold_boxes)}/{len(box_files)} boxes are cold")
    else:
        print("  None found")
    
    print(f"\n--- LAND COVER ANOMALIES ---")
    if lc_anomalies:
        for bf, info in lc_anomalies:
            print(f"  {bf}: neg={info['negative_count']}, >21={info['above_21_count']}, "
                  f"nan={info['nan_count']}, unique={info['unique_values']}")
    else:
        print("  None found (all values in 0-21 range, no NaN)")
    
    print(f"\n--- INPUT OUT OF RANGE ---")
    if input_anomalies:
        for bf, oor in input_anomalies:
            for var, info in oor.items():
                if info['below_pct'] > 0.1 or info['above_pct'] > 0.1:
                    print(f"  {bf}: {var} below={info['below_pct']:.2f}% above={info['above_pct']:.2f}%  "
                          f"data={info['data_range']}  train={info['train_range']}")
    else:
        print("  All inputs within training range")


def cmd_compare(args):
    """Compare channel attention between a bad and good box."""
    mt, device = load_model(args.model_folder)
    input_vars = mt.get_input_variable_names()
    min_inputs, max_inputs, _, _ = get_norm_params(mt)
    
    print(f"{'='*70}")
    print(f"  BAD BOX: {args.bad_box}")
    print(f"{'='*70}")
    bad_ds = xr.open_dataset(args.bad_box)
    bad_result = infer_box(mt, device, bad_ds, input_vars)
    bad_oor = check_input_ranges(bad_ds, input_vars, min_inputs, max_inputs)
    bad_lc = check_landcover(bad_ds)
    
    print(f"  Predicted temperature: mean={bad_result['temp_mean']:.1f}K, "
          f"min={bad_result['temp_min']:.1f}K, max={bad_result['temp_max']:.1f}K")
    if 'era5_skt' in bad_ds:
        era5 = float(bad_ds['era5_skt'].values.mean())
        print(f"  Raw ERA5 skt: {era5:.1f}K  (delta = {bad_result['temp_mean'] - era5:.1f}K)")
    
    print(f"\n  Normalized input per channel:")
    for i, var in enumerate(input_vars):
        ch = bad_result['normalized_input'][i]
        print(f"    [{i:2d}] {var:40s}  min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}")
    
    if bad_lc:
        print(f"\n  Land cover: unique={bad_lc['unique_values']}, neg={bad_lc['negative_count']}, "
              f">21={bad_lc['above_21_count']}, nan={bad_lc['nan_count']}")
    if bad_oor:
        print(f"  Out of range: {bad_oor}")
    
    print(f"\n{'='*70}")
    print(f"  GOOD BOX: {args.good_box}")
    print(f"{'='*70}")
    good_ds = xr.open_dataset(args.good_box)
    good_result = infer_box(mt, device, good_ds, input_vars)
    
    print(f"  Predicted temperature: mean={good_result['temp_mean']:.1f}K, "
          f"min={good_result['temp_min']:.1f}K, max={good_result['temp_max']:.1f}K")
    if 'era5_skt' in good_ds:
        era5 = float(good_ds['era5_skt'].values.mean())
        print(f"  Raw ERA5 skt: {era5:.1f}K  (delta = {good_result['temp_mean'] - era5:.1f}K)")
    
    print(f"\n  Normalized input per channel:")
    for i, var in enumerate(input_vars):
        ch = good_result['normalized_input'][i]
        print(f"    [{i:2d}] {var:40s}  min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}")
    
    # ATTENTION COMPARISON
    print(f"\n{'='*70}")
    print(f"  CHANNEL ATTENTION COMPARISON")
    print(f"{'='*70}")
    
    for layer_idx in sorted(bad_result['attention_weights'].keys()):
        bad_w = bad_result['attention_weights'][layer_idx]
        good_w = good_result['attention_weights'][layer_idx]
        diff = bad_w - good_w
        
        n_ch = len(bad_w)
        bad_suppressed = np.sum(bad_w < 0.3)
        good_suppressed = np.sum(good_w < 0.3)
        
        print(f"\n  Decoder attention layer {layer_idx} ({n_ch} channels):")
        print(f"    BAD:  mean={bad_w.mean():.4f}  suppressed(<0.3)={bad_suppressed}/{n_ch}")
        print(f"    GOOD: mean={good_w.mean():.4f}  suppressed(<0.3)={good_suppressed}/{n_ch}")
        print(f"    Max |diff|: {np.max(np.abs(diff)):.4f}")
        
        # Top channels with biggest suppression difference
        top_idx = np.argsort(diff)[:5]  # most suppressed in bad vs good
        print(f"    Top 5 channels MORE SUPPRESSED in bad box:")
        for idx in top_idx:
            print(f"      ch{idx:3d}: bad={bad_w[idx]:.4f}  good={good_w[idx]:.4f}  diff={diff[idx]:+.4f}")
    
    bad_ds.close()
    good_ds.close()


def cmd_landcover(args):
    """Probe land cover values across dates."""
    all_unique = set()
    per_file = {}
    
    dirs_to_check = [args.box_dir]
    if args.box_dir2:
        dirs_to_check.append(args.box_dir2)
    
    for d in dirs_to_check:
        box_files = sorted([f for f in os.listdir(d) if f.endswith('.nc')])
        print(f"\n{'='*70}")
        print(f"  Checking {len(box_files)} boxes in {d}")
        print(f"{'='*70}")
        
        dir_unique = set()
        dir_anomalies = []
        
        for bf in box_files:
            path = os.path.join(d, bf)
            ds = xr.open_dataset(path)
            lc_info = check_landcover(ds)
            ds.close()
            
            if not lc_info:
                continue
            
            vals = set(lc_info['unique_values'])
            dir_unique.update(vals)
            all_unique.update(vals)
            
            # Flag anything weird
            if lc_info['negative_count'] > 0 or lc_info['nan_count'] > 0 or lc_info['above_21_count'] > 0:
                dir_anomalies.append((bf, lc_info))
            
            # Flag fractional values (should be integer)
            non_int = [v for v in lc_info['unique_values'] if v != int(v)]
            if non_int:
                dir_anomalies.append((bf, f"NON-INTEGER values: {non_int}"))
        
        print(f"  All unique values in this dir: {sorted(dir_unique)}")
        if dir_anomalies:
            print(f"  ANOMALIES:")
            for bf, info in dir_anomalies:
                print(f"    {bf}: {info}")
        else:
            print(f"  No anomalies detected")
    
    print(f"\n{'='*70}")
    print(f"  ALL UNIQUE LAND COVER VALUES: {sorted(all_unique)}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose hallucination in LST model")
    subparsers = parser.add_subparsers(dest='command')
    
    # scan
    p_scan = subparsers.add_parser('scan', help='Scan boxes for anomalies')
    p_scan.add_argument('--model-folder', required=True)
    p_scan.add_argument('--box-dir', required=True)
    
    # compare
    p_cmp = subparsers.add_parser('compare', help='Compare bad vs good box')
    p_cmp.add_argument('--model-folder', required=True)
    p_cmp.add_argument('--bad-box', required=True)
    p_cmp.add_argument('--good-box', required=True)
    
    # landcover
    p_lc = subparsers.add_parser('landcover', help='Probe land cover values')
    p_lc.add_argument('--box-dir', required=True)
    p_lc.add_argument('--box-dir2', required=False, default=None)
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'landcover':
        cmd_landcover(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
