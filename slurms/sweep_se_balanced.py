import torch, numpy as np, xarray as xr, os
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

input_vars = mt.get_input_variable_names()
era5_idx = input_vars.index('era5_skt')

in_dir = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04'
out_dir = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04'

# SE England bounds (OSGB)
se_x = (400000, 700000)
se_y = (50000, 250000)
# Wider region for supplementing warm boxes
wide_x = (300000, 700000)
wide_y = (50000, 400000)

# First pass: catalogue all boxes with coordinates and est temp
print("Cataloguing boxes...", flush=True)
all_boxes = []
for fn in sorted(os.listdir(out_dir)):
    if not fn.endswith('.nc'): continue
    di = xr.open_dataset(os.path.join(in_dir, fn))
    do = xr.open_dataset(os.path.join(out_dir, fn))
    est = do['model_output'].values
    for i in range(est.shape[0]):
        x = float(di.SW_corner_x.values[i])
        y = float(di.SW_corner_y.values[i])
        m = float(np.nanmean(est[i]))
        in_se = se_x[0] <= x < se_x[1] and se_y[0] <= y < se_y[1]
        in_wide = wide_x[0] <= x < wide_x[1] and wide_y[0] <= y < wide_y[1]
        all_boxes.append({
            'file': fn, 'idx': i, 'est': m,
            'x': x, 'y': y, 'in_se': in_se, 'in_wide': in_wide
        })
    di.close(); do.close()

se_cold = [b for b in all_boxes if b['in_se'] and b['est'] < 280]
se_warm = [b for b in all_boxes if b['in_se'] and b['est'] >= 290]
wide_warm = [b for b in all_boxes if b['in_wide'] and not b['in_se'] and b['est'] >= 290]

print(f"SE boxes: {len(se_cold)} cold (<280K), {len(se_warm)} warm (>=290K)", flush=True)
print(f"Wide region extra warm: {len(wide_warm)}", flush=True)

# Sample: up to 25 cold from SE, up to 25 warm (SE first, then wide)
import random
random.seed(42)
sample_cold = random.sample(se_cold, min(25, len(se_cold)))
warm_pool = se_warm + wide_warm
sample_warm = random.sample(warm_pool, min(25, len(warm_pool)))

sample = sample_cold + sample_warm
print(f"Testing {len(sample_cold)} cold + {len(sample_warm)} warm = {len(sample)} boxes", flush=True)

# Sweep
sweep_vals = np.linspace(0.0, 1.0, 21)

# Norm params for converting output
np_dict = mt.normalisation_parameters
out_min = np_dict['min_output']
out_max = np_dict['max_output']
era5_min = np_dict['min_inputs']['era5_skt']
era5_max = np_dict['max_inputs']['era5_skt']

print(f"\nERA5 range: {era5_min:.1f}K - {era5_max:.1f}K", flush=True)
print(f"Output range: {out_min:.1f}K - {out_max:.1f}K", flush=True)

# Header
print(f"\ntype  est_K   x       y       region  actual_e5  thresh_e5  jump_K   curve_summary", flush=True)

# Group by file for efficient loading
from collections import defaultdict
by_file = defaultdict(list)
for b in sample:
    by_file[b['file']].append(b)

with torch.no_grad():
    for fn, boxes in by_file.items():
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
            inp_t = torch.tensor(inp).unsqueeze(0).to(device)
            actual_era5 = float(inp[era5_idx].mean())

            outputs = []
            for sv in sweep_vals:
                inp_sw = inp_t.clone()
                inp_sw[0, era5_idx, :, :] = sv
                enc, skips = mt.encoder(inp_sw)
                out = float(mt.decoder(enc, skips).mean())
                outputs.append(out)

            outputs_k = np.array(outputs) * (out_max - out_min) + out_min

            # Find largest jump
            diffs = np.diff(outputs_k)
            max_jump_idx = np.argmax(np.abs(diffs))
            threshold_norm = (sweep_vals[max_jump_idx] + sweep_vals[max_jump_idx + 1]) / 2
            threshold_k = threshold_norm * (era5_max - era5_min) + era5_min
            jump_k = diffs[max_jump_idx]

            tag = 'COLD' if b['est'] < 280 else 'WARM'
            region = 'SE' if b['in_se'] else 'WIDE'

            # Compact curve: show output at 5 key sweep points
            key_idx = [0, 5, 10, 15, 20]
            curve = ' '.join([f"{outputs_k[j]:.0f}" for j in key_idx])

            print(f"{tag:4s}  {b['est']:6.1f}  {b['x']:7.0f} {b['y']:7.0f}  {region:4s}  "
                  f"{actual_era5:.3f}      {threshold_k:.1f}K    {jump_k:+6.1f}K  [{curve}]", flush=True)

        ds.close()

print("\nDone.", flush=True)
