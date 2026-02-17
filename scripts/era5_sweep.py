import torch, numpy as np, xarray as xr
from cae_tools.models.unet import UNET
from cae_tools.models.ds_dataset import DSDataset

print("Loading...", flush=True)
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
era5_idx = input_vars.index('era5_skt')

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

# Get norm params for ERA5 to convert back to Kelvin
norm_params = mt.normalisation_parameters
if isinstance(norm_params, dict):
    era5_min = norm_params['min_inputs']['era5_skt']
    era5_max = norm_params['max_inputs']['era5_skt']
else:
    era5_min = norm_params[0]['era5_skt']
    era5_max = norm_params[1]['era5_skt']
out_min = norm_params[2] if not isinstance(norm_params, dict) else norm_params['min_output']
out_max = norm_params[3] if not isinstance(norm_params, dict) else norm_params['max_output']

print(f"ERA5 range: {era5_min:.1f}K - {era5_max:.1f}K", flush=True)
print(f"Output range: {out_min:.1f}K - {out_max:.1f}K", flush=True)

means = [float(np.nanmean(est[i])) for i in range(est.shape[0])]
indices = list(range(len(means)))
indices.sort(key=lambda b: means[b])
# 2 cold, 2 warm
test_indices = indices[:2] + indices[-2:]

# Sweep ERA5 normalized from 0.0 to 1.0 in steps
sweep_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"\nSweeping ERA5 for 2 cold + 2 warm boxes", flush=True)
print(f"ERA5 normalized value -> output in Kelvin", flush=True)

header = "type     est    actual"
for sv in sweep_vals:
    era5_k = era5_min + sv * (era5_max - era5_min)
    header += f"  {era5_k:5.0f}K"
print(header, flush=True)

with torch.no_grad():
    for i in test_indices:
        inp, _, _ = score_ds[i]
        inp_t = torch.tensor(inp).unsqueeze(0).to(device)
        m = means[i]
        tag = 'COLD' if m < 280 else 'WARM'

        # Get actual ERA5 normalized value
        actual_era5 = float(inp[era5_idx].mean())

        row = f"{tag}  {m:7.1f}K  {actual_era5:.3f}"
        for sv in sweep_vals:
            inp_sw = inp_t.clone()
            inp_sw[0, era5_idx, :, :] = sv
            enc, skips = mt.encoder(inp_sw)
            out = float(mt.decoder(enc, skips).mean())
            out_k = out_min + out * (out_max - out_min)
            row += f"  {out_k:6.1f}"
        print(row, flush=True)

print("\nDone.", flush=True)
