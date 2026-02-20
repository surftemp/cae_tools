"""
Trace through decoder sub-operations at ERA5 = 0.52 vs 0.53
to find exactly where the smooth encoder output becomes a step.
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

# Load normalisation
norm_path = os.path.join(model_path, 'normalisation.weights')
with open(norm_path) as f:
    norm_data = json.load(f)
min_inputs = norm_data['min_inputs']
max_inputs = norm_data['max_inputs']

channel_names = [
    'land_cover', 'albedo_monthly_climatology_means', 'elevation', 'era5_skt',
    'sin_doy', 'cos_doy', 'slope_magnitude', 'slope_direction',
    'urban_area', 'suburban_area', 'pixel_st_hot_pattern', 'pixel_st_cold_pattern'
]

# Load and assemble cold box
ds_in = xr.open_dataset('/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04/input_010.nc')
ds_out = xr.open_dataset('/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04/input_010.nc')
preds = ds_out['model_output'].values
cold_idx = int(np.argmin([np.nanmean(preds[i]) for i in range(preds.shape[0])]))

box_channels = []
for ch_name in channel_names:
    raw = ds_in[ch_name].values
    if raw.ndim == 1:
        ch_data = np.full((1, 100, 100), float(raw[cold_idx]))
    else:
        ch_data = raw[cold_idx]
    mn, mx = min_inputs[ch_name], max_inputs[ch_name]
    rng = mx - mn
    ch_data = (ch_data - mn) / rng if rng > 0 else np.zeros_like(ch_data)
    box_channels.append(ch_data)

box = np.concatenate(box_channels, axis=0)[np.newaxis, :, :, :]

# List decoder layers for reference
print("=== Decoder layer structure ===")
for i, layer in enumerate(mt.decoder.decoder_conv):
    print(f"  [{i}] {layer}")
print(f"  Attention layers: {len(mt.decoder.attention_layers)}")
print()

# For each ERA5 value, run encoder then manually step through decoder
for era5_val in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]:
    test_box = box.copy()
    test_box[0, 3, :, :] = era5_val
    x_input = torch.tensor(test_box, dtype=torch.float32)

    with torch.no_grad():
        # Run encoder
        encoded, skips = mt.encoder(x_input)

        # Decoder FC: latent -> unflatten
        x = mt.decoder.decoder_lin(encoded)
        x = mt.decoder.unflatten(x)
        after_unflatten = x.numpy().copy()

        # Reverse skips
        x_skip = skips[::-1]

        print(f"ERA5={era5_val:.2f} | Unflatten: mean={after_unflatten.mean():+.6f} std={after_unflatten.std():.6f}")

        skip_idx = 0
        stage = 0
        for i, layer in enumerate(mt.decoder.decoder_conv):
            x = layer(x)
            layer_name = type(layer).__name__

            if isinstance(layer, torch.nn.ConvTranspose2d) and skip_idx < len(x_skip):
                after_convt = x.numpy().copy()

                # Attention
                if mt.decoder.use_attention:
                    attention = mt.decoder.attention_layers[skip_idx](x)
                    x = x * attention
                after_attn = x.numpy().copy()

                # Skip join (concat mode)
                skip = x_skip[skip_idx]
                skip_np = skip.numpy().copy()
                if mt.decoder.skip_mode == 'concat':
                    x = torch.cat((x, skip), 1)
                else:
                    x = x + skip
                after_skip = x.numpy().copy()

                print(f"  Stage {stage} ConvT:  mean={after_convt.mean():+.6f} std={after_convt.std():.6f} zero%={(after_convt==0).mean()*100:.1f}%  shape={list(after_convt.shape)}")
                print(f"  Stage {stage} Attn:   mean={after_attn.mean():+.6f} std={after_attn.std():.6f} zero%={(after_attn==0).mean()*100:.1f}%")
                print(f"  Stage {stage} Skip:   mean={skip_np.mean():+.6f} std={skip_np.std():.6f} zero%={(skip_np==0).mean()*100:.1f}%")
                print(f"  Stage {stage} Joined: mean={after_skip.mean():+.6f} std={after_skip.std():.6f} zero%={(after_skip==0).mean()*100:.1f}%  shape={list(after_skip.shape)}")

                skip_idx += 1
                stage += 1

            elif isinstance(layer, torch.nn.BatchNorm2d):
                after_bn = x.numpy().copy()
                print(f"  Stage {stage-1} BN:     mean={after_bn.mean():+.6f} std={after_bn.std():.6f} neg%={(after_bn<0).mean()*100:.1f}%")

            elif isinstance(layer, torch.nn.ReLU):
                after_relu = x.numpy().copy()
                print(f"  Stage {stage-1} ReLU:   mean={after_relu.mean():+.6f} std={after_relu.std():.6f} zero%={(after_relu==0).mean()*100:.1f}%")

            elif isinstance(layer, torch.nn.Dropout):
                pass  # skip dropout in eval mode

            elif isinstance(layer, torch.nn.ConvTranspose2d) and skip_idx >= len(x_skip):
                # Final ConvT - no skip
                after_final = x.numpy().copy()
                print(f"  FINAL ConvT: mean={after_final.mean():+.6f} std={after_final.std():.6f} min={after_final.min():.4f} max={after_final.max():.4f}  shape={list(after_final.shape)}")

        # Apply sigmoid
        out = torch.sigmoid(x)
        temp_k = 250.26 + out.mean().item() * 118.62
        print(f"  Sigmoid -> Output: mean={out.mean().item():.6f}  Temp={temp_k:.1f}K")
        print()
