#!/bin/bash
# =============================================================================
# Experiment: STANDARD RESIDUAL UNET WITH AUGMENTATION
#
# Same as standard residual UNet but with data augmentation enabled:
#   - Random H-flip (p=0.5) with slope_direction correction
#   - Random V-flip (p=0.5) with slope_direction correction
#   - Applied on-GPU per batch, no changes to .pt files
#
# Does NOT require a layer definitions JSON (architecture is self-contained)
# Uses preprocessed .pt files
# =============================================================================

nrEpochs=3500
learningRate=0.0003
lambda_pearson=0.0005
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
method="unet"
checkpointInterval=500
baseChannels=64

# ---- Paths ----
databasePath="database_pB_standard_aug.db"
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"

# ---- Preprocessed data files ----
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8.pt"
testFile="${PREPROCESSED_DIR}/test_v8.pt"

if [ ! -f "$trainFile" ] || [ ! -f "$testFile" ]; then
    echo "Error: Preprocessed data not found in $PREPROCESSED_DIR"
    exit 1
fi

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
modelFolder="${MODELS_DIR}/model_pB_standard_aug_$hash"

echo "==========================================="
echo "Experiment: STANDARD RESIDUAL UNET + AUGMENTATION"
echo "Model: $modelFolder"
echo "Epochs: $nrEpochs"
echo "Architecture: standard"
echo "Base channels: $baseChannels"
echo "Output activation: none (unconstrained)"
echo "Augmentation: ON (H-flip + V-flip with slope_dir correction)"
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
    --architecture=standard \
    --base-channels="$baseChannels" \
    --output-activation=none \
    --augment \
    --slope-direction-channel=7

echo "==========================================="
echo "Training complete: $modelFolder"
echo "==========================================="
