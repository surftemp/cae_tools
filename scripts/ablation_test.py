import torch, numpy as np, xarray as xr, sys
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

print("Loading data...", flush=True)
ds_in = xr.open_dataset('/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04/input_010.nc')
est = xr.open_dataset('/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04/input_010.nc')['model_output'].values
input_vars = mt.get_input_variable_names()
era5_idx = input_vars.index('era5_skt')
print(f"ERA5 is input channel {era5_idx}", flush=True)

case_dim = ds_in[input_vars[0]].dims[0]
for var in input_vars:
    if ds_in[var].dims == (case_dim,):
        vals = ds_in[var].values
        y_dim, x_dim = ds_in.dims['y'], ds_in.dims['x']
        ds_in[var] = xr.DataArray(
            np.broadcast_to(vals[:, None, None, None], (vals.shape[0], 1, y_dim, x_dim)),
            dims=(case_dim, 'channel', 'y', 'x'))

print("Creating dataset...", flush=True)
score_ds = DSDataset(ds_in, input_vars, input_vars[0], normalise_in=True)
score_ds.set_normalisation_parameters(mt.normalisation_parameters)

# Pick 5 coldest and 5 warmest
means = [float(np.nanmean(est[i])) for i in range(est.shape[0])]
indices = list(range(len(means)))
indices.sort(key=lambda b: means[b])
test_indices = indices[:5] + indices[-5:]

print("\nRunning ablation...", flush=True)
print("type     est    normal  zero_lat  d(lat)  zero_era5  d(era5)", flush=True)

with torch.no_grad():
    for i in test_indices:
        inp, _, _ = score_ds[i]
        inp_t = torch.tensor(inp).unsqueeze(0).to(device)
        m = means[i]
        tag = 'COLD' if m < 280 else 'WARM'

        # Normal
        enc, skips = mt.encoder(inp_t)
        out_n = float(mt.decoder(enc, [s.clone() for s in skips]).mean())

        # Zero latent, keep skips
        enc_z = torch.zeros_like(enc)
        out_zl = float(mt.decoder(enc_z, [s.clone() for s in skips]).mean())

        # Zero ERA5 in input
        inp_ne = inp_t.clone()
        inp_ne[0, era5_idx, :, :] = 0.0
        enc2, skips2 = mt.encoder(inp_ne)
        out_ne = float(mt.decoder(enc2, skips2).mean())

        print(f'{tag}  {m:7.1f}K  {out_n:.4f}   {out_zl:.4f}   {out_zl-out_n:+.4f}    {out_ne:.4f}    {out_ne-out_n:+.4f}', flush=True)

print("\nDone.", flush=True)
