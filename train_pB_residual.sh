#!/bin/bash
# =============================================================================
# Experiment: RESIDUAL DELTA LEARNING
#   output = ERA5 + model_prediction(all 12 inputs)
#
# Key architectural choices:
#   - predict_delta: model learns Δ = LST - ERA5, scoring adds ERA5 back
#   - output_activation=tanh: [-1,1] output, gradient=1 at operating point
#   - latent_activation=none: fixes 98.5% dead latent (no ReLU before latent)
#   - skip_mode=add: prevents BN+ReLU from suppressing decoder channels
#
# Uses pB_spec_add.json (halved decoder channels for additive skips)
# Uses preprocessed delta .pt files (train_v8_delta.pt / test_v8_delta.pt)
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
layerDefinitionsPath="pB_spec_add.json"
databasePath="database_pB_residual.db"
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"

# ---- Preprocessed DELTA data files ----
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8_delta.pt"
testFile="${PREPROCESSED_DIR}/test_v8_delta.pt"

if [ ! -f "$trainFile" ] || [ ! -f "$testFile" ]; then
    echo "Error: Delta preprocessed data not found in $PREPROCESSED_DIR"
    echo "Run submit_preprocess_delta.slurm first"
    exit 1
fi

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
modelFolder="${MODELS_DIR}/model_pB_residual_$hash"

echo "==========================================="
echo "Experiment: RESIDUAL DELTA LEARNING"
echo "Model: $modelFolder"
echo "Epochs: $nrEpochs"
echo "Output activation: tanh"
echo "Latent activation: none"
echo "Skip mode: add"
echo "Predict delta: yes (ERA5 added back at scoring)"
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
    --layer-definitions-path="$layerDefinitionsPath" \
    --database-path="$databasePath" \
    --checkpoint-interval="$checkpointInterval" \
    --skip-mode=add \
    --latent-activation=none \
    --output-activation=tanh \
    --predict-delta \
    --delta-reference-channel=era5_skt

echo "==========================================="
echo "Training complete: $modelFolder"
echo "==========================================="
