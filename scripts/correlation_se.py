import torch, numpy as np, xarray as xr, os, random
from cae_tools.models.unet import UNET
from cae_tools.models.ds_dataset import DSDataset

print("Loading model...", flush=True)
mt = UNET()
mt.load('/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_BxW3tfkE/checkpoint_epoch_2500')
mt.encoder.eval()
device = torch.device('cpu')
mt.encoder.to(device)

input_vars = mt.get_input_variable_names()
in_dir = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04'
out_dir = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04'

# SE England bounds
se_x = (400000, 700000)
se_y = (50000, 250000)

# First pass: find SE England boxes and their est temps
print("Finding SE England boxes...", flush=True)
se_boxes = []
for fn in sorted(os.listdir(out_dir)):
    if not fn.endswith('.nc'): continue
    di = xr.open_dataset(os.path.join(in_dir, fn))
    do = xr.open_dataset(os.path.join(out_dir, fn))
    est = do['model_output'].values
    for i in range(est.shape[0]):
        x = float(di.SW_corner_x.values[i])
        y = float(di.SW_corner_y.values[i])
        if se_x[0] <= x < se_x[1] and se_y[0] <= y < se_y[1]:
            m = float(np.nanmean(est[i]))
            se_boxes.append({'file': fn, 'idx': i, 'est': m})
    di.close(); do.close()

print(f"Found {len(se_boxes)} SE England boxes", flush=True)
cold_se = [b for b in se_boxes if b['est'] < 280]
warm_se = [b for b in se_boxes if b['est'] >= 290]
print(f"  Cold (<280K): {len(cold_se)}, Warm (>=290K): {len(warm_se)}", flush=True)

# Sample up to 200 boxes (mix of cold and warm)
sample = cold_se + random.sample(warm_se, min(150, len(warm_se)))
random.shuffle(sample)

# Group by file for efficient loading
from collections import defaultdict
by_file = defaultdict(list)
for b in sample:
    by_file[b['file']].append(b)

print(f"Sampling {len(sample)} boxes from {len(by_file)} files...", flush=True)

latents = []
input_ch_means = []
est_vals = []

with torch.no_grad():
    for fn, boxes in by_file.items():
        print(f"  {fn} ({len(boxes)} boxes)...", flush=True)
        ds = xr.open_dataset(os.path.join(in_dir, fn))
        case_dim = ds[input_vars[0]].dims[0]
        for var in input_vars:
            if ds[var].dims == (case_dim,):
                vals = ds[var].values
                y_dim, x_dim = ds.dims['y'], ds.dims['x']
                ds[var] = xr.DataArray(
                    np.broadcast_to(vals[:, None, None, None], (vals.shape[0], 1, y_dim, x_dim)),
                    dims=(case_dim, 'channel', 'y', 'x'))
        sds = DSDataset(ds, input_vars, input_vars[0], normalise_in=True)
        sds.set_normalisation_parameters(mt.normalisation_parameters)

        for b in boxes:
            inp, _, _ = sds[b['idx']]
            input_ch_means.append([float(inp[ch].mean()) for ch in range(12)])
            inp_t = torch.tensor(inp).unsqueeze(0).to(device)
            encoded, _ = mt.encoder(inp_t)
            latents.append(encoded.cpu().numpy().flatten())
            est_vals.append(b['est'])
        ds.close()

latents = np.array(latents)
input_ch_means = np.array(input_ch_means)
est_vals = np.array(est_vals)

print(f"\nTotal samples: {len(latents)}", flush=True)

# Active dims
active_mask = (latents > 0).mean(axis=0) > 0.1
active_dims = np.where(active_mask)[0]
print(f"Active latent dims: {len(active_dims)}")
print(f"Indices: {active_dims}")

# Correlation table
print("\nPearson correlation: active_latent_dim vs input_channel_mean (SE England)")
header = '        '
for vn in input_vars:
    header += f'{vn[:8]:>10s}'
header += '     est'
print(header)

for di in active_dims:
    row = f'dim{di:4d}'
    for vi in range(12):
        s = latents[:, di].std()
        if s < 1e-8:
            r = 0
        else:
            r = float(np.corrcoef(latents[:, di], input_ch_means[:, vi])[0, 1])
            if np.isnan(r): r = 0
        row += f'{r:10.3f}'
    # Also correlate with est
    s = latents[:, di].std()
    if s < 1e-8:
        r = 0
    else:
        r = float(np.corrcoef(latents[:, di], est_vals)[0, 1])
        if np.isnan(r): r = 0
    row += f'{r:8.3f}'
    print(row)

# Also: correlation of each input channel mean with est
print("\nDirect correlation: input_channel_mean vs model_est (SE England)")
for vi, vn in enumerate(input_vars):
    r = float(np.corrcoef(input_ch_means[:, vi], est_vals)[0, 1])
    if np.isnan(r): r = 0
    print(f"  {vn:45s} r={r:+.3f}")

print("\nDone.", flush=True)
