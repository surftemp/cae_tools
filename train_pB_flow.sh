#!/bin/bash
# =============================================================================
# Experiment: FLOW MATCHING UNET
#
# Conditional flow matching (rectified flow) for LST downscaling:
#   - Same UNet backbone as standard architecture (residual blocks, GroupNorm, etc.)
#   - Timestep conditioning injected into every ResBlock
#   - Trains velocity field prediction (not direct regression)
#   - Inference via 4-step Euler integration from noise to prediction
#   - ~4x inference cost vs single-pass UNet
#
# Does NOT require a layer definitions JSON (architecture is self-contained)
# Uses preprocessed .pt files
# =============================================================================

nrEpochs=3500
learningRate=0.0003
lambda_pearson=0.0
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
method="unet"
checkpointInterval=500
baseChannels=64
flowSteps=4

# ---- Paths ----
databasePath="database_pB_flow.db"
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"

# ---- Preprocessed data files ----
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8_clean.pt"
testFile="${PREPROCESSED_DIR}/test_v8_clean.pt"

if [ ! -f "$trainFile" ] || [ ! -f "$testFile" ]; then
    echo "Error: Preprocessed data not found in $PREPROCESSED_DIR"
    exit 1
fi

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
modelFolder="${MODELS_DIR}/model_pB_flow_$hash"

echo "==========================================="
echo "Experiment: FLOW MATCHING UNET"
echo "Model: $modelFolder"
echo "Epochs: $nrEpochs"
echo "Architecture: flow_matching"
echo "Base channels: $baseChannels"
echo "Flow steps: $flowSteps"
echo "Augmentation: ON"
echo "==========================================="

train_cae \
    --train-inputs ${trainFile} \
    --test-inputs ${testFile} \
    --preprocessed \
    --model-folder "$modelFolder" \
    --nr-epochs="$nrEpochs" \
    --learning-rate="$learningRate" \
    --lambda-pearson="$lambda_pearson" \
    --weight-decay="$weight_decay" \
    --dropout-rate="$dropout_rate" \
    --batch-size="$batchSize" \
    --method="$method" \
    --database-path="$databasePath" \
    --checkpoint-interval="$checkpointInterval" \
    --architecture=flow_matching \
    --base-channels="$baseChannels" \
    --flow-steps="$flowSteps" \
    --augment \
    --slope-direction-channel=7

echo "==========================================="
echo "Training complete: $modelFolder"
echo "==========================================="
