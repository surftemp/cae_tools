#!/bin/bash
# =============================================================================
# Training script using PREPROCESSED .pt files
# Much faster startup, lower memory requirements
# =============================================================================

# ---- Hyperparameters ----
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

# ---- Generate unique model ID ----
hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"
echo "Model will be saved to: /gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash"

# ---- Train the model ----
train_cae \
    --train-inputs ${trainFile} \
    --test-inputs ${testFile} \
    --preprocessed \
    --model-folder "/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash" \
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

echo "Training complete!"
echo "Model saved to: /gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_$hash"
