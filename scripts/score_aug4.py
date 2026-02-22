import os, sys, json
import xarray as xr
import numpy as np

sys.path.insert(0, '/home/users/shaerdan/cae_tools_pB/src')
from cae_tools.models.unet import UNET

INPUT_DIR  = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04'
OUTPUT_DIR = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04_standard_aug_ep170'
CHECKPOINT = '/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_standard_aug_ff1bc219/checkpoint_best_test_mse'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model once
print("Loading model...", flush=True)
mt = UNET()
mt.load(CHECKPOINT)
input_variable_names = mt.get_input_variable_names()
print(f"Model loaded. Variables: {input_variable_names}", flush=True)

files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith('.nc'))
total_cold = 0
total_boxes = 0

for i, fname in enumerate(files):
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"[{i+1}/{len(files)}] skipping {fname} (exists)", flush=True)
        # still count for final tally
        ds = xr.open_dataset(out_path)
        est = ds['model_output'].values
        means = est.reshape(est.shape[0], -1).mean(axis=1)
        total_cold += int((means < 280).sum())
        total_boxes += len(means)
        ds.close()
        continue

    print(f"[{i+1}/{len(files)}] {fname}", flush=True)

    # Exact replication of apply_cae.py body from here:
    input_ds = [xr.open_dataset(os.path.join(INPUT_DIR, fname))]
    case_dimension = input_ds[0][input_variable_names[0]].dims[0]
    score_ds = input_ds[0]

    # broadcast scalar inputs to spatial — exact copy from apply_cae.py
    for var in input_variable_names:
        dims = score_ds[var].dims
        if dims == (case_dimension,):
            y_dim, x_dim = score_ds.dims['y'], score_ds.dims['x']
            original_values = score_ds[var].values
            expanded_values = np.broadcast_to(
                original_values[:, np.newaxis, np.newaxis, np.newaxis],
                (original_values.shape[0], 1, y_dim, x_dim))
            expanded_var = xr.DataArray(
                expanded_values,
                coords={case_dimension: score_ds[case_dimension],
                        'channel': [0],
                        'y': np.arange(y_dim),
                        'x': np.arange(x_dim)},
                dims=(case_dimension, 'channel', 'y', 'x'))
            score_ds[var] = expanded_var

    mt.apply(score_ds, input_variable_names, 'model_output')
    score_ds.to_netcdf(out_path)

    est = score_ds['model_output'].values
    means = est.reshape(est.shape[0], -1).mean(axis=1)
    cold = int((means < 280).sum())
    total_cold += cold
    total_boxes += len(means)
    print(f"  cold: {cold}/{len(means)}, running total: {total_cold}/{total_boxes}", flush=True)
    score_ds.close()

print(f"\nFinal: Cold (<280K): {total_cold}/{total_boxes} ({100*total_cold/total_boxes:.1f}%)")