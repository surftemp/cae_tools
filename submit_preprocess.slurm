#!/bin/bash
#SBATCH --job-name=preprocess_v8
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --partition=short-serial
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH -o /home/users/shaerdan/cae_tools_pB/logs/preprocess_%j.out
#SBATCH -e /home/users/shaerdan/cae_tools_pB/logs/preprocess_%j.err

# ============================================================================
# Preprocess v8 netCDF data to .pt format
# Run this ONCE, then use preprocessed files for all training runs
# ============================================================================

source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Input variables (same as training)
INPUT_VARS="land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction urban_area suburban_area pixel_st_hot_pattern pixel_st_cold_pattern"

# Output directory
OUTPUT_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
mkdir -p $OUTPUT_DIR

# ---- Preprocess Training Data ----
echo ""
echo "Preprocessing training data..."
preprocess_data \
    --input-files /gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/train/train_*.nc \
    --output-file ${OUTPUT_DIR}/train_v8.pt \
    --input-variables ${INPUT_VARS} \
    --output-variable ST_slices

# ---- Preprocess Test Data ----
echo ""
echo "Preprocessing test data..."
preprocess_data \
    --input-files /gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/test/test_*.nc \
    --output-file ${OUTPUT_DIR}/test_v8.pt \
    --input-variables ${INPUT_VARS} \
    --output-variable ST_slices

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "Output files:"
ls -lh ${OUTPUT_DIR}/*.pt
echo "End time: $(date)"
echo "=========================================="
