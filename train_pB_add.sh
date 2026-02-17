#!/bin/bash
# =============================================================================
# Experiment A: ADDITIVE SKIP CONNECTIONS
# Tests whether concat+BN+ReLU in decoder suppresses ERA5 in skip channels.
# Uses additive (true residual) skip connections instead of concatenation.
# All other hyperparameters match baseline pB for fair comparison.
# Requires pB_spec_add.json (halved decoder input channels)
# =============================================================================

nrEpochs=3500
learningRate=0.001
lambda_pearson=0.0005
weight_decay=1e-5
dropout_rate=0.3
batchSize=512
method="unet"
checkpointInterval=500

# ---- Paths ----
layerDefinitionsPath="pB_spec_add.json"
databasePath="database_pB_add_p3dropout.db"
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
modelFolder="${MODELS_DIR}/model_pB_add_$hash"

echo "=========================================="
echo "Experiment A: ADDITIVE SKIP CONNECTIONS"
echo "Model: $modelFolder"
echo "Epochs: $nrEpochs"
echo "Skip mode: add"
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
    --method="$method" \
    --layer-definitions-path="$layerDefinitionsPath" \
    --database-path="$databasePath" \
    --checkpoint-interval="$checkpointInterval" \
    --skip-mode=add

echo "=========================================="
echo "Training complete: $modelFolder"
echo "=========================================="
