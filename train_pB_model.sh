#!/bin/bash
# =============================================================================
# Training script for pB model with DATA SAMPLING
# Use with pB_stable branch of cae_tools
# =============================================================================

# ---- Hyperparameters (exact match to pB model) ----
nrEpochs=6000
learningRate=0.001
lambda_pearson=0.0005
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
fcSize=3200
latentSize=800
method="unet"

# ---- Sampling configuration ----
TRAIN_FRACTION=1  # 50% ≈ 141 files ≈ 47GB

# ---- Paths ----
layerDefinitionsPath="pB_spec.json"
databasePath="database_pB_stable.db"

# ---- Data folders ----
trainFolder="/gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/train/"
testFolder="/gws/nopw/j04/eocis_chuk/downscaling_full/10km_v8/test/"

# ---- Collect training files with sampling ----
allTrainFiles=(${trainFolder}*.nc)
totalFiles=${#allTrainFiles[@]}
sampleSize=$(printf "%.0f" $(echo "$totalFiles * $TRAIN_FRACTION" | bc))

echo "Total training files: $totalFiles"
echo "Sampling $sampleSize files (${TRAIN_FRACTION} fraction)"

# Shuffle with fixed seed for reproducibility, then take sample
mapfile -t trainPaths < <(printf '%s\n' "${allTrainFiles[@]}" | shuf --random-source=<(yes 42) -n $sampleSize)
trainPathsString="${trainPaths[@]}"

echo "Training files selected: ${#trainPaths[@]}"

# ---- Collect test files (use all) ----
testPaths=()
for file in ${testFolder}*.nc; do
    testPaths+=("$file")
done
testPathsString="${testPaths[@]}"

# ---- Generate unique model ID ----
hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"
echo "Model will be saved to: /gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash"

# ---- Input variables (12 channels) ----
INPUT_VARS="land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction urban_area suburban_area pixel_st_hot_pattern pixel_st_cold_pattern"

# ---- Train the model ----
train_cae \
    --train-inputs ${trainPathsString} \
    --test-inputs ${testPathsString} \
    --model-folder "/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash" \
    --input-variables ${INPUT_VARS} \
    --output-variable="ST_slices" \
    --nr-epochs="$nrEpochs" \
    --learning-rate="$learningRate" \
    --lambda-pearson="$lambda_pearson" \
    --weight-decay="$weight_decay" \
    --dropout-rate="$dropout_rate" \
    --batch-size="$batchSize" \
    --fc-size="$fcSize" \
    --latent-size="$latentSize" \
    --method="$method" \
    --layer-definitions-path="$layerDefinitionsPath" \
    --database-path="$databasePath"

# ---- Apply model to training data ----
apply_cae ${trainPathsString} \
    "/gws/nopw/j04/eocis_chuk/shaerdan/scores/train_scores_pB_$hash.nc" \
    --model-folder="/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash" \
    --input-variables ${INPUT_VARS} \
    --prediction-variable="hires_estimate"

# ---- Apply model to test data ----
apply_cae ${testPathsString} \
    "/gws/nopw/j04/eocis_chuk/shaerdan/scores/test_scores_pB_$hash.nc" \
    --model-folder="/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash" \
    --input-variables ${INPUT_VARS} \
    --prediction-variable="hires_estimate"

echo "Training complete!"
echo "Model saved to: /gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash"