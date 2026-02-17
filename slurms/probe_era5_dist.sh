#!/bin/bash
#SBATCH --job-name=era5_dist
#SBATCH -o era5_dist.out
#SBATCH -e era5_dist.err
#SBATCH --mem=80G
#SBATCH -t 0:30:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=standard

conda activate pyt_cae_tools

python -u << 'PYEOF'
import torch, numpy as np

print('Loading training data...', flush=True)
data = torch.load('/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/train_v8.pt',
                   map_location='cpu', weights_only=False)

is_normalized = data['normalized']
norm_params = data['normalisation_parameters']
input_vars = data['input_variables']
print(f'Input variables: {input_vars}', flush=True)
print(f'Normalized: {is_normalized}', flush=True)
print(f'Norm params keys: {list(norm_params.keys())}', flush=True)

era5_idx = input_vars.index('era5_skt')
print(f'ERA5 channel index: {era5_idx}', flush=True)

# Extract ERA5 value per box (scalar, just take one pixel)
era5_vals = data['inputs'][:, era5_idx, 0, 0].numpy()
# Extract output mean per box
out_means = data['outputs'][:, 0, :, :].mean(dim=(1,2)).numpy()

print(f'\nERA5 (normalized={is_normalized}):')
print(f'  min={era5_vals.min():.4f}, max={era5_vals.max():.4f}')
print(f'  mean={era5_vals.mean():.4f}, std={era5_vals.std():.4f}')
print(f'  median={np.median(era5_vals):.4f}')

print(f'\nOutput means:')
print(f'  min={out_means.min():.4f}, max={out_means.max():.4f}')
print(f'  mean={out_means.mean():.4f}')

# Histogram of ERA5 in 20 bins
print(f'\nERA5 histogram (20 bins):')
counts, edges = np.histogram(era5_vals, bins=20)
for i in range(len(counts)):
    bar = '#' * (counts[i] * 50 // counts.max())
    print(f'  [{edges[i]:.3f} - {edges[i+1]:.3f}]: {counts[i]:6d} {bar}')

# Check the critical region around 0.5 normalized (if normalized)
print(f'\nFine histogram around 0.45-0.65:')
mask = (era5_vals >= 0.45) & (era5_vals <= 0.65)
counts2, edges2 = np.histogram(era5_vals[mask], bins=20)
for i in range(len(counts2)):
    bar = '#' * (counts2[i] * 50 // max(counts2.max(), 1))
    print(f'  [{edges2[i]:.4f} - {edges2[i+1]:.4f}]: {counts2[i]:5d} {bar}')

# Correlation: ERA5 vs output mean
r = np.corrcoef(era5_vals, out_means)[0,1]
print(f'\nCorrelation ERA5 vs output mean: r={r:.4f}')

# Scatter: bin ERA5 into 20 bins, show mean output per bin
print(f'\nERA5 bin -> mean output:')
for i in range(len(edges)-1):
    m = (era5_vals >= edges[i]) & (era5_vals < edges[i+1])
    if m.sum() > 0:
        print(f'  ERA5 [{edges[i]:.3f}-{edges[i+1]:.3f}]: n={m.sum():5d}, out_mean={out_means[m].mean():.4f}, out_std={out_means[m].std():.4f}')

print('\nDone.', flush=True)
PYEOF
