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

print("Loading data...", flush=True)
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

print("Creating dataset...", flush=True)
score_ds = DSDataset(ds_in, input_vars, input_vars[0], normalise_in=True)
score_ds.set_normalisation_parameters(mt.normalisation_parameters)

means = [float(np.nanmean(est[i])) for i in range(est.shape[0])]
indices = list(range(len(means)))
indices.sort(key=lambda b: means[b])
# 3 cold, 3 warm
test_indices = indices[:3] + indices[-3:]

print(f"\nAblating each of 12 input channels for 3 cold + 3 warm boxes", flush=True)
print(f"Values shown are delta from normal (in sigmoid 0-1 scale)", flush=True)
print(f"1 sigmoid unit = ~119K, so 0.01 = ~1.2K", flush=True)

# Header
header = "type     est    "
for vi, vn in enumerate(input_vars):
    header += f"{vn[:7]:>9s}"
print(header, flush=True)

with torch.no_grad():
    for i in test_indices:
        inp, _, _ = score_ds[i]
        inp_t = torch.tensor(inp).unsqueeze(0).to(device)
        m = means[i]
        tag = 'COLD' if m < 280 else 'WARM'

        # Normal
        enc, skips = mt.encoder(inp_t)
        out_n = float(mt.decoder(enc, [s.clone() for s in skips]).mean())

        row = f"{tag}  {m:7.1f}K "
        for ch in range(12):
            inp_abl = inp_t.clone()
            inp_abl[0, ch, :, :] = 0.0
            enc_a, skips_a = mt.encoder(inp_abl)
            out_a = float(mt.decoder(enc_a, skips_a).mean())
            delta = out_a - out_n
            row += f"  {delta:+.4f}"
        print(row, flush=True)

print("\nDone.", flush=True)
