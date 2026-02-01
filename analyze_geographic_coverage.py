import xarray as xr
import numpy as np
import glob

files = sorted(glob.glob('/gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/train/train_*.nc'))

print(f'Analyzing {len(files)} files...')

# Collect one representative coordinate per file (the 40km region center)
region_coords = []
boxes_per_region = []

for f in files:
    ds = xr.open_dataset(f)
    if 'SW_corner_x' in ds:
        x = ds['SW_corner_x'].values
        y = ds['SW_corner_y'].values
        # Use mean as region center
        region_coords.append((np.mean(x), np.mean(y)))
        boxes_per_region.append(len(x))
    ds.close()

region_coords = np.array(region_coords)
boxes_per_region = np.array(boxes_per_region)

# UK approximate bounds in BNG:
# Scotland: Y > 500000
# England/Wales: Y < 500000

scotland_mask = region_coords[:, 1] > 500000
england_mask = region_coords[:, 1] <= 500000

print()
print('Geographic distribution:')
print(f'  Scotland (Y > 500000): {scotland_mask.sum()} regions, {boxes_per_region[scotland_mask].sum()} boxes')
print(f'  England/Wales:         {england_mask.sum()} regions, {boxes_per_region[england_mask].sum()} boxes')

print()
print('Y coordinate ranges:')
print(f'  Min Y: {region_coords[:, 1].min():.0f}')
print(f'  Max Y: {region_coords[:, 1].max():.0f}')

print()
print('Boxes per region stats:')
print(f'  Min: {boxes_per_region.min()}')
print(f'  Max: {boxes_per_region.max()}')
print(f'  Mean: {boxes_per_region.mean():.1f}')