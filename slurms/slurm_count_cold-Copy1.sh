#!/bin/bash
#SBATCH --job-name=count_cold
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=eocis_chuk
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/count_cold_%j.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/count_cold_%j.err
# =============================================================================
# Count cold boxes (<280K) on a stitched UK map NetCDF.
# Runs on lst_model_estimate variable from collect_scored_dataset output.
#
# Usage (typically submitted with --dependency=afterok:<collect_job_id>):
#   sbatch --dependency=afterok:<COLLECT_JOB_ID> \
#       slurm_count_cold.sh <MAP_NC> <LABEL>
#
# Example:
#   sbatch --dependency=afterok:123457 \
#       slurm_count_cold.sh \
#       /gws/nopw/j04/eocis_chuk/shaerdan/models/scores/uk_maps/2023-08-04_standard_aug_ep170.nc \
#       "standard_aug ep170"
# =============================================================================

MAP_NC=$1
LABEL=$2

if [ -z "$MAP_NC" ]; then
    echo "Error: missing MAP_NC argument"
    echo "Usage: sbatch slurm_count_cold.sh MAP_NC [LABEL]"
    exit 1
fi

if [ ! -f "$MAP_NC" ]; then
    echo "Error: map file not found: $MAP_NC"
    exit 1
fi

set --
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"

echo "Map:   $MAP_NC"
echo "Label: $LABEL"
echo ""

python3 - << PYEOF
import xarray as xr
import numpy as np

map_nc = "$MAP_NC"
label  = "$LABEL"

ds  = xr.open_dataset(map_nc, drop_variables=['time_bnds'])
est = ds['lst_model_estimate'].values.flatten()
est = est[~np.isnan(est)]

total = len(est)
cold  = int((est < 280).sum())

print(f"Model:  {label}")
print(f"File:   {map_nc}")
print(f"")
print(f"Total pixels (non-NaN): {total:,}")
print(f"Cold (<280K):           {cold:,}  ({100*cold/total:.2f}%)")
print(f"Est range:              [{est.min():.1f}K, {est.max():.1f}K]")
print(f"Est mean:               {est.mean():.1f}K")
print(f"")
bins = [0, 260, 270, 280, 290, 300, 310, 320, 330, 400]
hist, _ = np.histogram(est, bins=bins)
print("Distribution:")
for j in range(len(bins)-1):
    if hist[j] > 0:
        print(f"  {bins[j]:3d}-{bins[j+1]:3d}K: {hist[j]:8,}  ({100*hist[j]/total:.1f}%)")
ds.close()
PYEOF
