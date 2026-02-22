#!/bin/bash
# =============================================================================
# SELF-RESUBMITTING TRAINING SCRIPT FOR FLOW MATCHING (ORCHID, 24h limit)
#
# Same mechanics as train_pB_standard_aug_auto.sh, but EPOCHS_PER_JOB is lower
# because flow matching test evaluation runs Euler integration (4 forward passes
# per batch instead of 1), making test epochs ~3-4x slower than standard UNet.
#
# Usage (first run):  sbatch train_pB_flow_auto.sh
# Usage (continue):   MODEL_FOLDER=/path/to/model sbatch train_pB_flow_auto.sh
# =============================================================================

#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pB_flow_zscore_rm_coldpattern
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/pB_flow_zscore_rm_coldpattern_%j.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/pB_flow_zscore_rm_coldpattern_%j.err
#SBATCH --signal=TERM@120


# Load conda environment
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"
export PYTHONUNBUFFERED=1


# ============ CONFIGURATION ============
TOTAL_EPOCHS=3500
EPOCHS_PER_JOB=250          # Lower than standard UNet — test epoch is ~4x slower
                             # due to Euler integration in flow_matching_sample()
learningRate=0.0003
lambda_pearson=0.0
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
checkpointInterval=50       # More frequent checkpoints since jobs are shorter
baseChannels=64
flowSteps=4
databasePath="database_pB_flow.db"
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8_zscore_rm_coldpattern.pt"
testFile="${PREPROCESSED_DIR}/test_v8_zscore_rm_coldpattern.pt"
# =======================================

set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Architecture: flow_matching (Euler steps: $flowSteps)"
echo "==========================================="

if [ ! -f "$trainFile" ] || [ ! -f "$testFile" ]; then
    echo "ERROR: Preprocessed data not found in $PREPROCESSED_DIR"
    exit 1
fi

CONTINUE_FLAG=""
if [ -n "${MODEL_FOLDER:-}" ] && [ -f "${MODEL_FOLDER}/parameters.json" ]; then
    modelFolder="$MODEL_FOLDER"
    CONTINUE_FLAG="--continue-training"
    echo "Continuing training from: $modelFolder"
else
    hash=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
    modelFolder="${MODELS_DIR}/model_pB_flow_zscore_rm_coldpattern_${hash}"
    echo "Fresh training, model folder: $modelFolder"
fi

_resubmit() {
    echo "SIGTERM received - resubmitting with MODEL_FOLDER=$modelFolder"
    sbatch --export=MODEL_FOLDER="$modelFolder" $(realpath "$0")
    exit 0
}
trap '_resubmit' TERM

echo "Total target epochs: $TOTAL_EPOCHS"
echo "Epochs this job: $EPOCHS_PER_JOB"
echo "==========================================="

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
    --architecture=flow_matching \
    --base-channels="$baseChannels" \
    --flow-steps="$flowSteps" \
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
    EPOCHS_DONE=$(python3 -c "
import json
with open('${modelFolder}/history.json') as f:
    h = json.load(f)
print(h.get('nr_epochs', 0))
")
fi

echo "Epochs completed: $EPOCHS_DONE / $TOTAL_EPOCHS"

if [ "$EPOCHS_DONE" -ge "$TOTAL_EPOCHS" ]; then
    echo "Training COMPLETE. Model at: $modelFolder"
else
    echo "Resubmitting (${EPOCHS_DONE}/${TOTAL_EPOCHS} done)..."
    NEXT_JOB=$(MODEL_FOLDER="$modelFolder" sbatch \
        --dependency=afterok:$SLURM_JOB_ID \
        --export=MODEL_FOLDER="$modelFolder" \
        $(realpath "$0"))
    echo "Submitted: $NEXT_JOB — will continue from: $modelFolder"
fi

echo "Job finished: $(date)"
