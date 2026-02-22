#!/bin/bash
# =============================================================================
# SELF-RESUBMITTING: STANDARD RESIDUAL UNET + ON-THE-FLY AUGMENTATION
#
# Uses train_v8_clean.pt with --augment (D4: H-flip + V-flip + slope correction).
# Consistent with current running experiments.
#
# Test set is NOT augmented (never augment test data).
# =============================================================================

#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pB_standard_aug_zscore_nomonthly
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/pB_aug_zscore_nomonthly_%j.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/pB_aug_zscore_nomonthly_%j.err
#SBATCH --signal=TERM@120


# Load conda environment
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"
export PYTHONUNBUFFERED=1



TOTAL_EPOCHS=3500
learningRate=0.0003
lambda_pearson=0
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
checkpointInterval=500
baseChannels=64
databasePath="database_pB_standard_aug_zscore_nomonthly.db"
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8_zscore_nomonthly.pt"
testFile="${PREPROCESSED_DIR}/test_v8_zscore_nomonthly.pt"

set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "Start: $(date)  |  Target: $TOTAL_EPOCHS epochs"
echo "==========================================="

if [ ! -f "$trainFile" ]; then
    echo "ERROR: $trainFile not found."
    exit 1
fi

CONTINUE_FLAG=""
if [ -n "${MODEL_FOLDER:-}" ] && [ -f "${MODEL_FOLDER}/parameters.json" ]; then
    modelFolder="$MODEL_FOLDER"
    CONTINUE_FLAG="--continue-training"
    echo "Continuing from: $modelFolder"
else
    hash=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
    modelFolder="${MODELS_DIR}/model_pB_standard_aug_zscore_nomonthly_${hash}"
    echo "Fresh run: $modelFolder"
fi

_resubmit() {
    echo "SIGTERM received - resubmitting with MODEL_FOLDER=$modelFolder"
    sbatch --export=MODEL_FOLDER="$modelFolder" $(realpath "$0")
    exit 0
}
trap '_resubmit' TERM


# Disable -e around train_cae so we can inspect its exit code.
# Resubmit only happens on clean exit (0); any crash terminates the chain.
set +e
train_cae \
    --train-inputs ${trainFile} \
    --test-inputs ${testFile} \
    --preprocessed \
    --model-folder "$modelFolder" \
    --nr-epochs="$TOTAL_EPOCHS" \
    --learning-rate="$learningRate" \
    --lambda-pearson="$lambda_pearson" \
    --weight-decay="$weight_decay" \
    --dropout-rate="$dropout_rate" \
    --batch-size="$batchSize" \
    --method="unet" \
    --database-path="$databasePath" \
    --checkpoint-interval="$checkpointInterval" \
    --architecture=standard \
    --base-channels="$baseChannels" \
    --output-activation=none \
    --augment \
    --slope-direction-channel=7 \
    $CONTINUE_FLAG
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: train_cae exited with code $TRAIN_EXIT — chain terminated. Check logs."
    exit $TRAIN_EXIT
fi

EPOCHS_DONE=0
if [ -f "${modelFolder}/history.json" ]; then
    EPOCHS_DONE=$(python3 -c "import json; h=json.load(open('${modelFolder}/history.json')); print(h.get('nr_epochs',0))")
fi

echo "Epochs done: $EPOCHS_DONE / $TOTAL_EPOCHS"

if [ "$EPOCHS_DONE" -ge "$TOTAL_EPOCHS" ]; then
    echo "Training COMPLETE: $modelFolder"
else
    echo "Resubmitting..."
    sbatch --dependency=afterok:$SLURM_JOB_ID \
           --export=MODEL_FOLDER="$modelFolder" \
           $(realpath "$0")
fi
echo "Finished: $(date)"
