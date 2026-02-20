#!/bin/bash
#SBATCH --job-name=preprocess_delta
#SBATCH -o /home/users/shaerdan/cae_tools_pB/logs/preprocess_delta.out
#SBATCH -e /home/users/shaerdan/cae_tools_pB/logs/preprocess_delta.err
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=standard

# =============================================================================
# Preprocess training and test data for residual/delta learning:
#   target = T_LST - T_ERA5  (per pixel)
#   normalized to [-1, 1] (for tanh output activation)
#
# Produces:
#   train_v8_delta.pt  — training data with delta targets
#   test_v8_delta.pt   — test data using training normalization stats
# =============================================================================

conda activate pyt_cae_tools

PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
TRAIN_NC_DIR="/gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/train"
TEST_NC_DIR="/gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/test"

INPUT_VARS="land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction urban_area suburban_area pixel_st_hot_pattern pixel_st_cold_pattern"
OUTPUT_VAR="ST_slices"

echo "=========================================="
echo "Step 1: Preprocess TRAINING data (delta + tanh)"
echo "=========================================="

preprocess_data \
    --input-files ${TRAIN_NC_DIR}/*.nc \
    --output-file ${PREPROCESSED_DIR}/train_v8_delta.pt \
    --input-variables ${INPUT_VARS} \
    --output-variable ${OUTPUT_VAR} \
    --compute-delta \
    --delta-reference-variable era5_skt \
    --output-activation tanh

echo "=========================================="
echo "Step 2: Preprocess TEST data (using train stats)"
echo "=========================================="

preprocess_data \
    --input-files ${TEST_NC_DIR}/*.nc \
    --output-file ${PREPROCESSED_DIR}/test_v8_delta.pt \
    --input-variables ${INPUT_VARS} \
    --output-variable ${OUTPUT_VAR} \
    --compute-delta \
    --delta-reference-variable era5_skt \
    --output-activation tanh \
    --use-norm-from ${PREPROCESSED_DIR}/train_v8_delta.pt

echo "=========================================="
echo "Done. Files:"
echo "  ${PREPROCESSED_DIR}/train_v8_delta.pt"
echo "  ${PREPROCESSED_DIR}/test_v8_delta.pt"
echo "=========================================="
