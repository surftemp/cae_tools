#!/bin/bash
# =============================================================================
# Training script to reproduce pB model (bb8bb65f-7ccd-49d5-a9e0-c663abfbd068)
# Original training date: 2025-02-25
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

# ---- Paths ----
layerDefinitionsPath="pB_spec.json"
databasePath="database_pB_stable.db"

# ---- Data folders ----
trainFolder="/lustre_scratch/shaerdan/data_folder/train_v4/train/processed_train/"
testFolder="/lustre_scratch/shaerdan/data_folder/train_v4/test/processed_test/"

# ---- Collect training files ----
trainPaths=()
for file in ${trainFolder}*.nc; do
    trainPaths+=("$file")
done
trainPathsString="${trainPaths[@]}"

# ---- Collect test files ----
testPaths=()
for file in ${testFolder}*.nc; do
    testPaths+=("$file")
done
testPathsString="${testPaths[@]}"

# ---- Generate unique model ID ----
hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"
echo "Model will be saved to: /lustre_scratch/shaerdan/models/model_pB_$hash"

# ---- Input variables (12 channels - exact match to pB model) ----
INPUT_VARS="land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction urban_area suburban_area pixel_st_hot_pattern pixel_st_cold_pattern"

# ---- Train the model ----
train_cae \
    --train-inputs ${trainPathsString} \
    --test-inputs ${testPathsString} \
    --model-folder "/lustre_scratch/shaerdan/models/model_pB_$hash" \
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
    "/lustre_scratch/shaerdan/scores/train_scores_pB_$hash.nc" \
    --model-folder="/lustre_scratch/shaerdan/models/model_pB_$hash" \
    --input-variables ${INPUT_VARS} \
    --prediction-variable="hires_estimate"

# ---- Apply model to test data ----
apply_cae ${testPathsString} \
    "/lustre_scratch/shaerdan/scores/test_scores_pB_$hash.nc" \
    --model-folder="/lustre_scratch/shaerdan/models/model_pB_$hash" \
    --input-variables ${INPUT_VARS} \
    --prediction-variable="hires_estimate"

echo "Training complete!"
echo "Model saved to: /lustre_scratch/shaerdan/models/model_pB_$hash"
