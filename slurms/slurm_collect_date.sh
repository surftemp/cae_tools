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
# Usage:
#   sbatch slurm_collect_date.sh <DATE> <OUTPUT_DIR> <MAP_OUTPUT> <MODEL_ID> [--overwrite]
#
# Flags:
#   --overwrite   Regenerate map even if it already exists
#
# Default behaviour: skip collect+plot if map already exists,
#                    but cleanup of per-box files always runs if map exists.
# =============================================================================

DATE=$1
OUTPUT_DIR=$2
MAP_OUTPUT=$3
MODEL_ID=$4
OVERWRITE=${5:-""}

if [ -z "$DATE" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MAP_OUTPUT" ] || [ -z "$MODEL_ID" ]; then
    echo "Error: missing arguments"
    echo "Usage: sbatch slurm_collect_date.sh DATE OUTPUT_DIR MAP_OUTPUT MODEL_ID [--overwrite]"
    exit 1
fi

mkdir -p $(dirname $MAP_OUTPUT)

echo "=========================================="
echo "Date:      $DATE"
echo "Box dir:   $OUTPUT_DIR"
echo "Map out:   $MAP_OUTPUT"
echo "Model ID:  $MODEL_ID"
echo "Overwrite: ${OVERWRITE:-no}"
echo "Start:     $(date)"
echo "=========================================="

set --
source ~/miniforge3/bin/activate

if [ -f "$MAP_OUTPUT" ] && [ "$OVERWRITE" != "--overwrite" ]; then
    echo "Map already exists, skipping collect+plot: $MAP_OUTPUT"
else
    # Check box files are present
    n_boxes=$(ls $OUTPUT_DIR/input_*.nc 2>/dev/null | wc -l)
    echo "Found $n_boxes scored box files"
    if [ "$n_boxes" -eq 0 ]; then
        echo "Error: no scored box files found in $OUTPUT_DIR"
        exit 1
    fi

    # ---- Step 1: Collect into UK map ----
    conda activate downscaling_env

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
    conda deactivate
fi

# ---- Step 3: Clean up per-box intermediates ----
# Always runs if box dir exists - map is confirmed present at this point
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Cleaning up per-box files in $OUTPUT_DIR..."
    rm -rf $OUTPUT_DIR
    echo "Cleanup done."
fi

echo ""
echo "=========================================="
echo "Collect complete: $MAP_OUTPUT"
echo "End: $(date)"
echo "=========================================="
