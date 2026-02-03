#!/bin/bash
# =============================================================================
# Training script using PREPROCESSED .pt files
# Much faster startup, lower memory requirements
#
# Usage:
#   Fresh training:      ./train_pB_model_preprocessed.sh
#   Continue training:   CONTINUE_MODEL=model_pB_XXXXXXXX ./train_pB_model_preprocessed.sh
# =============================================================================

# ---- Hyperparameters ----
nrEpochs=3500
learningRate=0.001
lambda_pearson=0.0005
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
fcSize=3200
latentSize=800
method="unet"
checkpointInterval=500

# ---- Paths ----
layerDefinitionsPath="pB_spec.json"
databasePath="database_pB_stable.db"
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"

# ---- Preprocessed data files ----
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8.pt"
testFile="${PREPROCESSED_DIR}/test_v8.pt"

# Check files exist
if [ ! -f "$trainFile" ]; then
    echo "Error: Training file not found: $trainFile"
    echo "Run submit_preprocess.slurm first!"
    exit 1
fi

if [ ! -f "$testFile" ]; then
    echo "Error: Test file not found: $testFile"
    echo "Run submit_preprocess.slurm first!"
    exit 1
fi

# ---- Check if continuing or starting fresh ----
if [ -n "$CONTINUE_MODEL" ]; then
    # Continue from existing model
    modelFolder="${MODELS_DIR}/${CONTINUE_MODEL}"
    
    if [ ! -d "$modelFolder" ]; then
        echo "Error: Model folder not found: $modelFolder"
        exit 1
    fi
    
    echo "=========================================="
    echo "CONTINUING training from: $modelFolder"
    echo "Additional epochs: $nrEpochs"
    echo "Checkpoint interval: $checkpointInterval"
    echo "=========================================="
    
    train_cae \
        --train-inputs ${trainFile} \
        --test-inputs ${testFile} \
        --preprocessed \
        --continue-training \
        --model-folder "$modelFolder" \
        --nr-epochs="$nrEpochs" \
        --learning-rate="$learningRate" \
        --batch-size="$batchSize" \
        --database-path="$databasePath" \
        --checkpoint-interval="$checkpointInterval"
else
    # Fresh training - generate new model ID
    hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
    modelFolder="${MODELS_DIR}/model_pB_$hash"
    
    echo "=========================================="
    echo "FRESH training"
    echo "Model ID: $hash"
    echo "Model folder: $modelFolder"
    echo "Epochs: $nrEpochs"
    echo "Checkpoint interval: $checkpointInterval"
    echo "=========================================="
    
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
        --fc-size="$fcSize" \
        --latent-size="$latentSize" \
        --method="$method" \
        --layer-definitions-path="$layerDefinitionsPath" \
        --database-path="$databasePath" \
        --checkpoint-interval="$checkpointInterval"
fi

echo "=========================================="
echo "Training complete!"
echo "Model saved to: $modelFolder"
echo "=========================================="
