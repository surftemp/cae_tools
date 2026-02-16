#!/bin/bash
# =============================================================================
# Experiment: CONV BOTTLENECK (proper UNET with conv bridge, no FC layers)
# Same other hyperparameters as pB FC model for fair comparison
# =============================================================================

nrEpochs=3500
learningRate=0.001
lambda_pearson=0.0005
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
method="unet"
checkpointInterval=500

# ---- Paths ----
layerDefinitionsPath="pB_spec.json"
databasePath="database_pB_conv.db"
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
modelFolder="${MODELS_DIR}/model_pB_conv_$hash"

echo "=========================================="
echo "Experiment: CONV BOTTLENECK"
echo "Model: $modelFolder"
echo "Epochs: $nrEpochs"
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
    --bottleneck-type=conv

echo "=========================================="
echo "Training complete: $modelFolder"
echo "=========================================="
