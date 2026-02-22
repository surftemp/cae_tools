#!/bin/bash
#SBATCH --job-name=collect_date
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=eocis_chuk
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/collect_date_%j.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/collect_date_%j.err
# =============================================================================
# Collect scored boxes into a stitched UK-wide map, plot PNG, clean up boxes.
#
# Usage (typically submitted with --dependency=afterok:<array_job_id>):
#   sbatch --dependency=afterok:<SCORE_JOB_ID> \
#       slurm_collect_date.sh <DATE> <OUTPUT_DIR> <MAP_OUTPUT> <MODEL_ID>
#
# Example:
#   sbatch --dependency=afterok:123456 \
#       slurm_collect_date.sh \
#       2023-08-04 \
#       /gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04_standard_aug_ep170 \
#       /gws/nopw/j04/eocis_chuk/shaerdan/models/scores/uk_maps/2023-08-04_standard_aug_ep170.nc \
#       model_pB_standard_aug_ep170
# =============================================================================

DATE=$1
OUTPUT_DIR=$2        # directory containing scored input_*.nc box files
MAP_OUTPUT=$3        # path for final stitched UK map .nc
MODEL_ID=$4

if [ -z "$DATE" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MAP_OUTPUT" ] || [ -z "$MODEL_ID" ]; then
    echo "Error: missing arguments"
    echo "Usage: sbatch slurm_collect_date.sh DATE OUTPUT_DIR MAP_OUTPUT MODEL_ID"
    exit 1
fi

mkdir -p $(dirname $MAP_OUTPUT)

echo "=========================================="
echo "Date:      $DATE"
echo "Box dir:   $OUTPUT_DIR"
echo "Map out:   $MAP_OUTPUT"
echo "Model ID:  $MODEL_ID"
echo "Start:     $(date)"
echo "=========================================="

# Check all box files are present
n_boxes=$(ls $OUTPUT_DIR/input_*.nc 2>/dev/null | wc -l)
echo "Found $n_boxes scored box files"
if [ "$n_boxes" -eq 0 ]; then
    echo "Error: no scored box files found in $OUTPUT_DIR"
    exit 1
fi

# ---- Step 1: Collect into UK map ----
set --   # clear $@ so conda activate doesn't pick up script args
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"

echo ""
echo "Running collect_scored_dataset..."
collect_scored_dataset \
    --grid-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK-GRID-100M-v1.0.nc \
    --landwater-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-LANDWATER-MERGED-2023-fv1.1.nc \
    --country-path /gws/nopw/j04/eocis_chuk/shaerdan/EOCIS-CHUK_GEOSPATIAL_INFORMATION-L4-COUNTRY-MERGED-2023-fv1.1.nc \
    --day $DATE \
    --input-path $OUTPUT_DIR/input_*.nc \
    --score-variable model_output \
    --output-path $MAP_OUTPUT \
    --downweight-border 10 \
    --include-variables era5_skt \
    --model-id $MODEL_ID \
    --era5-regrid-path /gws/nopw/j04/eocis_chuk/downscaling_full/era5land/11am_daily_files_chuk

if [ ! -f "$MAP_OUTPUT" ]; then
    echo "Error: collect_scored_dataset did not produce $MAP_OUTPUT"
    exit 1
fi

echo "UK map saved: $MAP_OUTPUT"

# ---- Step 2: Plot PNG ----
conda deactivate
conda activate netcdfexplorer_env

PNG_OUTPUT="${MAP_OUTPUT%.nc}.png"
echo ""
echo "Plotting PNG..."
bigplot \
    --input-path $MAP_OUTPUT \
    --input-variable lst_model_estimate \
    --flip \
    --legend-width 400 \
    --vmin 270 \
    --vmax 330 \
    --title "$MODEL_ID LST estimate for $DATE" \
    --plot-width 2048 \
    --output-path $PNG_OUTPUT

echo "PNG saved: $PNG_OUTPUT"

# ---- Step 3: Clean up per-box intermediates ----
# This keeps disk usage bounded - the stitched map is all we need going forward
conda deactivate
echo ""
echo "Cleaning up per-box files in $OUTPUT_DIR..."
rm -rf $OUTPUT_DIR
echo "Cleanup done."

echo ""
echo "=========================================="
echo "Collect complete: $MAP_OUTPUT"
echo "End: $(date)"
echo "=========================================="
