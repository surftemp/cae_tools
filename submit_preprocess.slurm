#!/bin/bash
#SBATCH --job-name=preprocess_v8
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH -o /home/users/shaerdan/cae_tools_pB/logs/preprocess_%j.out
#SBATCH -e /home/users/shaerdan/cae_tools_pB/logs/preprocess_%j.err

# ============================================================================
# Preprocess v8 netCDF data to .pt format
# 
# IMPORTANT: Test data must use TRAIN normalisation parameters!
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
#INPUT_VARS="land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction urban_area suburban_area pixel_st_hot_pattern #pixel_st_cold_pattern"
# Input variables (same as training)
INPUT_VARS="land_cover elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction urban_area suburban_area"


# Output directory
OUTPUT_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
mkdir -p $OUTPUT_DIR

# ============================================================================
# Step 1: Preprocess Training Data (computes its own normalisation stats)
# ============================================================================
echo ""
echo "Step 1: Preprocessing training data (zscore filter)..."
preprocess_data \
    --input-files /gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/train/train_*.nc \
    --output-file ${OUTPUT_DIR}/train_v8_zscore_nomonthly.pt \
    --input-variables ${INPUT_VARS} \
    --output-variable ST_slices \
    --filter-cloud zscore \
    --zscore-threshold -3.0

# ============================================================================
# Step 2: Preprocess Test Data (uses TRAIN normalisation stats + delta stats)
# ============================================================================
echo ""
echo "Step 2: Preprocessing test data (zscore filter using train stats)..."
preprocess_data \
    --input-files /gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/test/test_*.nc \
    --output-file ${OUTPUT_DIR}/test_v8_zscore_nomonthly.pt \
    --input-variables ${INPUT_VARS} \
    --output-variable ST_slices \
    --use-norm-from ${OUTPUT_DIR}/train_v8_zscore_nomonthly.pt \
    --filter-cloud zscore \
    --zscore-threshold -3.0

# ============================================================================
# Step 3: Preprocess Validation Data (uses TRAIN normalisation stats + delta stats)
# ============================================================================
# Uncomment if you have a separate validation set
 echo ""
 echo "Step 3: Preprocessing validation data (zscore filter using train stats)..."
 preprocess_data \
     --input-files /gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/val/val_*.nc \
     --output-file ${OUTPUT_DIR}/val_v8_zscore_nomonthly.pt \
     --input-variables ${INPUT_VARS} \
     --output-variable ST_slices \
     --use-norm-from ${OUTPUT_DIR}/train_v8_zscore_nomonthly.pt \
     --filter-cloud zscore \
     --zscore-threshold -3.0

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/*.pt
echo ""
echo "IMPORTANT: All files are normalized using TRAINING data statistics."
echo "End time: $(date)"
echo "=========================================="
