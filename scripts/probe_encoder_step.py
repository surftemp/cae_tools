"""
Probe where the ERA5 step function forms in the encoder.
Sweep ERA5 from 0 to 1 (normalized) on a cold box, 
record skip connection statistics at each encoder stage.
"""
import torch
import numpy as np
import xarray as xr
import sys
import os
import json

sys.path.insert(0, '/home/users/shaerdan/cae_tools_pB/src')
from cae_tools.models.unet import UNET

# Load model
model_path = '/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_BxW3tfkE/checkpoint_epoch_2500'
mt = UNET()
mt.load(model_path)
mt.encoder.eval()
mt.decoder.eval()

# Load normalisation parameters
norm_path = os.path.join(model_path, 'normalisation.weights')
with open(norm_path) as f:
    norm_data = json.load(f)

# Handle both dict and list format
if isinstance(norm_data, dict):
    min_inputs = norm_data['min_inputs']
    max_inputs = norm_data['max_inputs']
elif isinstance(norm_data, list):
    min_inputs = norm_data[0]
    max_inputs = norm_data[1]
else:
    raise ValueError(f"Unknown normalisation format: {type(norm_data)}")

print("Normalisation parameters loaded:")
if isinstance(min_inputs, dict):
    for k in min_inputs:
        print(f"  {k}: [{min_inputs[k]:.4f}, {max_inputs[k]:.4f}]")
else:
    print(f"  min_inputs: {min_inputs}")
    print(f"  max_inputs: {max_inputs}")

# The 12 model input channels in order
channel_names = [
    'land_cover', 'albedo_monthly_climatology_means', 'elevation', 'era5_skt',
    'sin_doy', 'cos_doy', 'slope_magnitude', 'slope_direction',
    'urban_area', 'suburban_area', 'pixel_st_hot_pattern', 'pixel_st_cold_pattern'
]

# Load raw NetCDF
input_file = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04/input_010.nc'
output_file = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04/input_010.nc'

ds_in = xr.open_dataset(input_file)
ds_out = xr.open_dataset(output_file)

# Find coldest box
preds = ds_out['model_output'].values
cold_idx = int(np.argmin([np.nanmean(preds[i]) for i in range(preds.shape[0])]))
print(f'\nUsing box {cold_idx}, predicted temp = {np.nanmean(preds[cold_idx]):.1f}K')

# Assemble and normalize the 12-channel input for this box
box_channels = []
for ch_name in channel_names:
    raw = ds_in[ch_name].values  # (400,) scalar or (400, 1, 100, 100) spatial
    if raw.ndim == 1:
        # Scalar per box - broadcast to (1, 100, 100)
        val = float(raw[cold_idx])
        ch_data = np.full((1, 100, 100), val)
    elif raw.ndim == 4:
        ch_data = raw[cold_idx]  # (1, 100, 100)
    else:
        raise ValueError(f"Unexpected shape for {ch_name}: {raw.shape}")
    
    # Normalize to [0, 1]
    if isinstance(min_inputs, dict):
        mn = min_inputs[ch_name]
        mx = max_inputs[ch_name]
    else:
        ch_idx = channel_names.index(ch_name)
        mn = min_inputs[ch_idx]
        mx = max_inputs[ch_idx]
    
    rng = mx - mn
    if rng > 0:
        ch_data = (ch_data - mn) / rng
    else:
        ch_data = np.zeros_like(ch_data)
    
    box_channels.append(ch_data)

# Stack to (1, 12, 100, 100)
box = np.concatenate(box_channels, axis=0)[np.newaxis, :, :, :]  # (1, 12, 100, 100)
print(f'Assembled input shape: {box.shape}')
print(f'ERA5 normalized value: {box[0, 3, :, :].mean():.4f}')
print(f'ERA5 spatially uniform: std={box[0, 3, :, :].std():.6f}')

# Verify: run full model and check output matches expected
with torch.no_grad():
    x_test = torch.tensor(box, dtype=torch.float32)
    encoded, skips_test = mt.encoder(x_test)
    decoded = mt.decoder(encoded, skips_test)
    verify_temp = 250.26 + decoded.mean().item() * 118.62
    print(f'Verification: model output = {verify_temp:.1f}K (should be ~{np.nanmean(preds[cold_idx]):.1f}K)')

# ========================================
# SWEEP ERA5 - track skip statistics
# ========================================
print()
era5_channel = 3
era5_values = np.linspace(0.0, 1.0, 21)

print(f'{"ERA5":>5} | {"Skip1 mean":>10} {"Skip1 std":>10} {"S1 zero%":>8} | '
      f'{"Skip2 mean":>10} {"Skip2 std":>10} {"S2 zero%":>8} | '
      f'{"Skip3 mean":>10} {"Skip3 std":>10} {"S3 zero%":>8} | '
      f'{"Temp(K)":>8}')
print('-' * 130)

for era5_val in era5_values:
    test_box = box.copy()
    test_box[0, era5_channel, :, :] = era5_val
    
    x = torch.tensor(test_box, dtype=torch.float32)
    
    # Run encoder layer by layer
    skips = []
    with torch.no_grad():
        h = x
        for layer in mt.encoder.encoder_cnn:
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU):
                skips.append(h.numpy().copy())
        
        # Full forward for output temp
        if mt.encoder.use_fc:
            flat = mt.encoder.flatten(h)
            enc = mt.encoder.encoder_lin(flat)
        else:
            enc = mt.encoder.bridge(h)
        
        skips_dec = [torch.tensor(s) for s in skips[:-1]]
        decoded = mt.decoder(enc, skips_dec)
        temp_k = 250.26 + decoded.mean().item() * 118.62
    
    s1, s2, s3 = skips[0], skips[1], skips[2]
    
    print(f'{era5_val:5.2f}  | {s1.mean():+10.5f} {s1.std():10.5f} {(s1==0).mean()*100:7.1f}% | '
          f'{s2.mean():+10.5f} {s2.std():10.5f} {(s2==0).mean()*100:7.1f}% | '
          f'{s3.mean():+10.5f} {s3.std():10.5f} {(s3==0).mean()*100:7.1f}% | '
          f'{temp_k:8.1f}')

# ========================================
# Fine sweep to find exact step location
# ========================================
print()
print('=== Fine sweep around step ===')
outputs_fine = []
for era5_val in np.linspace(0.0, 1.0, 101):
    test_box = box.copy()
    test_box[0, era5_channel, :, :] = era5_val
    x = torch.tensor(test_box, dtype=torch.float32)
    with torch.no_grad():
        enc, sk = mt.encoder(x)
        dec = mt.decoder(enc, sk)
        outputs_fine.append(dec.mean().item())

outputs_fine = np.array(outputs_fine)
diffs = np.abs(np.diff(outputs_fine))
step_idx = np.argmax(diffs)
step_era5 = step_idx / 100.0
print(f'Largest jump at ERA5_norm = {step_era5:.2f} -> {step_era5+0.01:.2f}')
print(f'Output: {outputs_fine[step_idx]:.4f} -> {outputs_fine[step_idx+1]:.4f}')
print(f'Temp:   {250.26 + outputs_fine[step_idx]*118.62:.1f}K -> {250.26 + outputs_fine[step_idx+1]*118.62:.1f}K')
