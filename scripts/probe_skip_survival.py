import torch, numpy as np, xarray as xr
from cae_tools.models.unet import UNET
from cae_tools.models.ds_dataset import DSDataset

print("Loading model...", flush=True)
mt = UNET()
mt.load('/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_BxW3tfkE/checkpoint_epoch_2500')
mt.encoder.eval()
mt.decoder.eval()
device = torch.device('cpu')
mt.encoder.to(device)
mt.decoder.to(device)

ds_in = xr.open_dataset('/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04/input_010.nc')
est = xr.open_dataset('/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04/input_010.nc')['model_output'].values
input_vars = mt.get_input_variable_names()

case_dim = ds_in[input_vars[0]].dims[0]
for var in input_vars:
    if ds_in[var].dims == (case_dim,):
        vals = ds_in[var].values
        y_dim, x_dim = ds_in.dims['y'], ds_in.dims['x']
        ds_in[var] = xr.DataArray(
            np.broadcast_to(vals[:, None, None, None], (vals.shape[0], 1, y_dim, x_dim)),
            dims=(case_dim, 'channel', 'y', 'x'))

score_ds = DSDataset(ds_in, input_vars, input_vars[0], normalise_in=True)
score_ds.set_normalisation_parameters(mt.normalisation_parameters)

means = [float(np.nanmean(est[i])) for i in range(est.shape[0])]
indices = list(range(len(means)))
indices.sort(key=lambda b: means[b])
test_indices = indices[:3] + indices[-3:]

print("Tracing skip channel survival through decoder BN+ReLU", flush=True)
print("After concat, first half = decoder channels, second half = skip channels", flush=True)
print("Checking what fraction of skip-side channels survive ReLU", flush=True)
print(flush=True)

with torch.no_grad():
    for i in test_indices:
        inp, _, _ = score_ds[i]
        inp_t = torch.tensor(inp).unsqueeze(0).to(device)
        m = means[i]
        tag = 'COLD' if m < 280 else 'WARM'

        encoded, skips = mt.encoder(inp_t)
        x = mt.decoder.decoder_lin(encoded)
        x = mt.decoder.unflatten(x)
        x_skip = skips[::-1]

        skip_idx = 0
        print(f"=== {tag} box est={m:.1f}K ===", flush=True)
        for layer in mt.decoder.decoder_conv:
            if isinstance(layer, torch.nn.ConvTranspose2d) and skip_idx < len(x_skip):
                x = layer(x)
                attn = mt.decoder.attention_layers[skip_idx](x)
                x = x * attn
                n_dec = x.shape[1]
                n_skip = x_skip[skip_idx].shape[1]
                x = torch.cat((x, x_skip[skip_idx]), 1)
                n_total = x.shape[1]
                # Record pre-BN+ReLU values for skip-side channels
                skip_pre = x[0, n_dec:, :, :].clone()
                skip_idx += 1
            elif isinstance(layer, torch.nn.ConvTranspose2d):
                x = layer(x)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                x = layer(x)
                # After BN, check skip-side channels
                skip_post_bn = x[0, n_dec:, :, :].clone()
            elif isinstance(layer, torch.nn.ReLU):
                x = layer(x)
                # After ReLU, check skip-side survival
                skip_post_relu = x[0, n_dec:, :, :].clone()

                # Per-channel survival: fraction of spatial pixels > 0
                alive_per_ch = (skip_post_relu > 0).float().mean(dim=(1,2))
                dead_channels = (alive_per_ch == 0).sum().item()
                total_ch = alive_per_ch.shape[0]

                # Compare energy: skip before BN vs after ReLU
                energy_pre = skip_pre.pow(2).mean().item()
                energy_post = skip_post_relu.pow(2).mean().item()
                ratio = energy_post / energy_pre if energy_pre > 0 else 0

                print(f"  Skip level {skip_idx}: {n_skip}ch, "
                      f"dead_channels={dead_channels}/{total_ch}, "
                      f"mean_survival={alive_per_ch.mean():.3f}, "
                      f"energy_ratio={ratio:.4f}", flush=True)
            elif isinstance(layer, torch.nn.Dropout):
                x = layer(x)

        print(flush=True)

print("Done.", flush=True)
