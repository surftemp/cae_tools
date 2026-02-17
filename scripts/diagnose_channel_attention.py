"""
Diagnostic script: Analyze channel attention weights during inference
to determine if ERA5 temperature channel is being suppressed on hallucinating dates.

Usage:
    python diagnose_channel_attention.py \
        --model-folder /gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_BxW3tfkE/checkpoint_epoch_2500 \
        --bad-box /path/to/model_inputs/2023-08-04/some_box_in_SE_england.nc \
        --good-box /path/to/model_inputs/2023-08-09/some_box_in_SE_england.nc

To get the boxes, first regenerate inputs for a bad and good date
:
    prepare_scoring_dataset \
        --grid-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK-GRID-100M-v1.0.nc \
        --land-cover-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-LANDCOVER-MERGED-2023-fv1.1.nc \
        --land-water-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-LANDWATER-MERGED-2023-fv1.1.nc \
        --built-area-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-BUILTAREA-MERGED-2023-fv1.1.nc \
        --elevation-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-ELEVATION-MERGED-2023-fv1.0.nc \
        --start-date 2023-08-04 --end-date 2023-08-04 \
        --box-stride 50 --output-folder debug_inputs_bad
        
    prepare_scoring_dataset \
        --grid-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK-GRID-100M-v1.0.nc \
        --land-cover-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-LANDCOVER-MERGED-2023-fv1.1.nc \
        --land-water-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-LANDWATER-MERGED-2023-fv1.1.nc \
        --built-area-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-BUILTAREA-MERGED-2023-fv1.1.nc \
        --elevation-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-ELEVATION-MERGED-2023-fv1.0.nc \
        --start-date 2023-08-09 --end-date 2023-08-09 \
        --box-stride 50 --output-folder debug_inputs_good

Then pick a box from the SE England region (eastings ~500000-600000, northings ~150000-250000).
"""

import argparse
import os
import json
import numpy as np
import torch
import xarray as xr

from cae_tools.models.unet import UNET
from cae_tools.models.ds_dataset import DSDataset


def analyze_box(model, score_ds, input_variable_names, device, label=""):
    """Run inference on a single box and extract attention weights + per-channel stats."""
    
    # Create dataset for this box
    ds = DSDataset(score_ds, input_variable_names, input_variable_names[0], normalise_in=model.normalise_input)
    ds.set_normalisation_parameters(model.normalisation_parameters)
    
    # Get the normalized input
    in_arr, _, _ = ds[0]
    input_tensor = torch.tensor(in_arr).unsqueeze(0).to(device)  # [1, 12, 100, 100]
    
    # Print per-channel input stats (normalized)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"\n  Normalized input stats per channel:")
    for i, var_name in enumerate(input_variable_names):
        ch = input_tensor[0, i]
        print(f"    [{i:2d}] {var_name:40s}  min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}")
    
    # Hook into attention layers to capture weights
    attention_weights = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            attention_weights.append({
                'layer': layer_idx,
                'weights': output.detach().cpu().numpy()  # [1, channels, 1, 1]
            })
        return hook_fn
    
    hooks = []
    for idx, attn_layer in enumerate(model.decoder.attention_layers):
        h = attn_layer.register_forward_hook(make_hook(idx))
        hooks.append(h)
    
    # Also hook into the decoder ConvTranspose layers to see pre/post attention
    conv_outputs = []
    def make_conv_hook(layer_idx):
        def hook_fn(module, input, output):
            conv_outputs.append({
                'layer': layer_idx,
                'output_mean': output.detach().cpu().mean(dim=(2,3)).numpy(),  # [1, channels]
                'output_std': output.detach().cpu().std(dim=(2,3)).numpy(),
            })
        return hook_fn
    
    conv_idx = 0
    for layer in model.decoder.decoder_conv:
        if isinstance(layer, torch.nn.ConvTranspose2d):
            h = layer.register_forward_hook(make_conv_hook(conv_idx))
            hooks.append(h)
            conv_idx += 1
    
    # Run inference
    model.encoder.eval()
    model.decoder.eval()
    with torch.no_grad():
        encoded, skip = model.encoder(input_tensor)
        
        if hasattr(model.decoder, 'decoder_lin') and model.decoder.decoder_lin is not None:
            decoded_flat = model.decoder.decoder_lin(encoded)
            bottleneck_reconstructed = model.decoder.unflatten(decoded_flat)
            print(f"\n  FC bottleneck: encoded dim={encoded.shape[1]}, "
                  f"reconstructed feature map mean={bottleneck_reconstructed.mean():.4f}, "
                  f"std={bottleneck_reconstructed.std():.4f}")
        
        output = model.decoder(encoded, skip)
    
    # Denormalize output to get temperature
    output_np = output.cpu().numpy()
    if isinstance(model.normalisation_parameters, dict):
        min_out = model.normalisation_parameters['min_output']
        max_out = model.normalisation_parameters['max_output']
    else:
        min_out = model.normalisation_parameters[2]
        max_out = model.normalisation_parameters[3]
    
    temp_output = min_out + output_np * (max_out - min_out)
    print(f"\n  Output temperature: min={temp_output.min():.1f}K, max={temp_output.max():.1f}K, "
          f"mean={temp_output.mean():.1f}K")
    
    # Print attention weights at each decoder level
    print(f"\n  Channel attention weights at each decoder level:")
    for aw in attention_weights:
        weights = aw['weights'][0, :, 0, 0]  # [channels]
        layer_idx = aw['layer']
        n_ch = len(weights)
        
        # Find suppressed and boosted channels
        suppressed = np.where(weights < 0.3)[0]
        boosted = np.where(weights > 0.7)[0]
        
        print(f"\n    Decoder attention layer {layer_idx} ({n_ch} channels):")
        print(f"      Weight range: [{weights.min():.4f}, {weights.max():.4f}], "
              f"mean={weights.mean():.4f}")
        print(f"      Suppressed (<0.3): {len(suppressed)}/{n_ch} channels")
        print(f"      Boosted (>0.7):    {len(boosted)}/{n_ch} channels")
        
        # For the first attention layer (256 channels), show distribution
        if layer_idx == 0:
            percentiles = np.percentile(weights, [5, 25, 50, 75, 95])
            print(f"      Percentiles [5,25,50,75,95]: {percentiles}")
    
    # Print skip connection stats
    print(f"\n  Skip connection stats:")
    for i, s in enumerate(skip):
        print(f"    Skip {i}: shape={list(s.shape)}, mean={s.mean():.4f}, std={s.std():.4f}")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return {
        'attention_weights': attention_weights,
        'output_temp': temp_output,
        'input_tensor': input_tensor.cpu().numpy(),
    }


def check_landcover_values(score_ds, input_variable_names):
    """Check for unexpected land cover values."""
    if 'land_cover' in input_variable_names:
        lc = score_ds['land_cover'].values
        unique_vals = np.unique(lc[~np.isnan(lc)])
        print(f"\n  Land cover unique values: {sorted(unique_vals.astype(int))}")
        print(f"  Land cover range: [{lc.min()}, {lc.max()}]")
        
        # Check for negative values or unexpected classes
        neg_count = np.sum(lc < 0)
        if neg_count > 0:
            print(f"  WARNING: {neg_count} pixels with NEGATIVE land cover values!")
        
        high_count = np.sum(lc > 21)
        if high_count > 0:
            print(f"  WARNING: {high_count} pixels with land cover > 21!")
        
        nan_count = np.sum(np.isnan(lc))
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN pixels in land cover!")


def check_input_ranges(score_ds, input_variable_names, norm_params):
    """Check if any input values fall outside training normalization range."""
    if isinstance(norm_params, dict):
        min_inputs = norm_params['min_inputs']
        max_inputs = norm_params['max_inputs']
    else:
        min_inputs = norm_params[0]
        max_inputs = norm_params[1]
    
    print(f"\n  Input range check (vs training normalization bounds):")
    for var_name in input_variable_names:
        if var_name not in score_ds:
            continue
        data = score_ds[var_name].values
        data_min = float(np.nanmin(data))
        data_max = float(np.nanmax(data))
        train_min = min_inputs[var_name]
        train_max = max_inputs[var_name]
        
        below = np.sum(data < train_min)
        above = np.sum(data > train_max)
        total = data.size
        
        status = "OK" if (below == 0 and above == 0) else "OUT OF RANGE"
        if status != "OK":
            print(f"    {var_name:40s}  {status}  "
                  f"below={below}({100*below/total:.2f}%)  above={above}({100*above/total:.2f}%)  "
                  f"data=[{data_min:.4f}, {data_max:.4f}]  train=[{train_min:.4f}, {train_max:.4f}]")
        else:
            print(f"    {var_name:40s}  {status}  "
                  f"data=[{data_min:.4f}, {data_max:.4f}]  train=[{train_min:.4f}, {train_max:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Diagnose channel attention on hallucinating boxes")
    parser.add_argument("--model-folder", required=True, help="Path to model checkpoint folder")
    parser.add_argument("--bad-box", required=False, help="Path to netCDF box from a hallucinating date")
    parser.add_argument("--good-box", required=False, help="Path to netCDF box from a normal date")
    parser.add_argument("--box-dir", required=False, help="Directory of boxes to scan for anomalies")
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    parameters_path = os.path.join(args.model_folder, "parameters.json")
    with open(parameters_path) as f:
        parameters = json.loads(f.read())
    
    mt = UNET()
    mt.load(args.model_folder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt.encoder.to(device)
    mt.decoder.to(device)
    
    input_variable_names = mt.get_input_variable_names()
    print(f"Input variables: {input_variable_names}")
    print(f"Device: {device}")
    
    if args.bad_box:
        print(f"\n{'#'*70}")
        print(f"  ANALYZING BAD BOX (hallucinating date)")
        print(f"{'#'*70}")
        score_ds = xr.open_dataset(args.bad_box)
        check_landcover_values(score_ds, input_variable_names)
        check_input_ranges(score_ds, input_variable_names, mt.normalisation_parameters)
        bad_results = analyze_box(mt, score_ds, input_variable_names, device, 
                                   label=f"BAD: {args.bad_box}")
    
    if args.good_box:
        print(f"\n{'#'*70}")
        print(f"  ANALYZING GOOD BOX (normal date)")
        print(f"{'#'*70}")
        score_ds = xr.open_dataset(args.good_box)
        check_landcover_values(score_ds, input_variable_names)
        check_input_ranges(score_ds, input_variable_names, mt.normalisation_parameters)
        good_results = analyze_box(mt, score_ds, input_variable_names, device,
                                    label=f"GOOD: {args.good_box}")
    
    if args.bad_box and args.good_box:
        print(f"\n{'#'*70}")
        print(f"  COMPARISON: BAD vs GOOD")
        print(f"{'#'*70}")
        
        for i in range(len(bad_results['attention_weights'])):
            bad_w = bad_results['attention_weights'][i]['weights'][0, :, 0, 0]
            good_w = good_results['attention_weights'][i]['weights'][0, :, 0, 0]
            diff = bad_w - good_w
            
            print(f"\n  Attention layer {i}:")
            print(f"    Max attention difference: {np.max(np.abs(diff)):.4f}")
            print(f"    Mean bad weights:  {bad_w.mean():.4f}")
            print(f"    Mean good weights: {good_w.mean():.4f}")
            
            # Find channels with biggest difference
            top_diff_idx = np.argsort(np.abs(diff))[-5:][::-1]
            print(f"    Top 5 channels with biggest attention difference:")
            for idx in top_diff_idx:
                print(f"      Channel {idx:3d}: bad={bad_w[idx]:.4f}, good={good_w[idx]:.4f}, diff={diff[idx]:+.4f}")
        
        print(f"\n  Temperature comparison:")
        print(f"    Bad box mean:  {bad_results['output_temp'].mean():.1f}K")
        print(f"    Good box mean: {good_results['output_temp'].mean():.1f}K")
    
    # Scan directory mode
    if args.box_dir:
        print(f"\n{'#'*70}")
        print(f"  SCANNING ALL BOXES IN {args.box_dir}")
        print(f"{'#'*70}")
        
        box_files = sorted([f for f in os.listdir(args.box_dir) if f.endswith('.nc')])
        print(f"Found {len(box_files)} boxes")
        
        cold_boxes = []
        for bf in box_files:
            path = os.path.join(args.box_dir, bf)
            score_ds = xr.open_dataset(path)
            
            # Quick inference to get output temperature
            ds = DSDataset(score_ds, input_variable_names, input_variable_names[0], 
                          normalise_in=mt.normalise_input)
            ds.set_normalisation_parameters(mt.normalisation_parameters)
            in_arr, _, _ = ds[0]
            input_tensor = torch.tensor(in_arr).unsqueeze(0).to(device)
            
            with torch.no_grad():
                encoded, skip = mt.encoder(input_tensor)
                output = mt.decoder(encoded, skip)
            
            output_np = output.cpu().numpy()
            if isinstance(mt.normalisation_parameters, dict):
                min_out = mt.normalisation_parameters['min_output']
                max_out = mt.normalisation_parameters['max_output']
            else:
                min_out = mt.normalisation_parameters[2]
                max_out = mt.normalisation_parameters[3]
            
            temp = min_out + output_np * (max_out - min_out)
            mean_temp = float(temp.mean())
            
            if mean_temp < 285:  # Suspiciously cold for summer
                cold_boxes.append((bf, mean_temp))
                
                # Check ERA5 input for this box
                if 'era5_skt' in input_variable_names:
                    era5_idx = input_variable_names.index('era5_skt')
                    era5_val = float(score_ds['era5_skt'].values.mean())
                    print(f"  COLD: {bf}  mean_output={mean_temp:.1f}K  era5_skt_input={era5_val:.1f}K  "
                          f"delta={mean_temp - era5_val:.1f}K")
        
        print(f"\n  Found {len(cold_boxes)} suspiciously cold boxes (mean < 285K)")


if __name__ == '__main__':
    main()
