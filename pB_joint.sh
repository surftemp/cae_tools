#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pB_joint
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/pB_joint_%j.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/pB_joint_%j.err
#SBATCH --signal=TERM@120
# =============================================================================
# SELF-RESUBMITTING: JOINT CN + UNET TRAINING
#
# CN corrects raw Landsat LST cloud contamination during training.
# Main UNET is unchanged at inference — plug directly into scoring pipeline.
#
# Uses zscore-cleaned .pt files with cold pattern removed.
# =============================================================================

source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"
export PYTHONUNBUFFERED=1

TOTAL_EPOCHS=3500
MODELS_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models"
PREPROCESSED_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8"
trainFile="${PREPROCESSED_DIR}/train_v8_zscore_rm_coldpattern.pt"
testFile="${PREPROCESSED_DIR}/test_v8_zscore_rm_coldpattern.pt"
databasePath="database_pB_joint.db"

set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "Start: $(date)  |  Target: $TOTAL_EPOCHS epochs"
echo "==========================================="

if [ ! -f "$trainFile" ]; then
    echo "ERROR: $trainFile not found."
    exit 1
fi

# ---- Resolve model folder ----
CONTINUE_FLAG=""
if [ -n "${MODEL_FOLDER:-}" ] && [ -f "${MODEL_FOLDER}/parameters.json" ]; then
    modelFolder="$MODEL_FOLDER"
    CONTINUE_FLAG="--continue-training"
    echo "Continuing from: $modelFolder"
else
    hash=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
    modelFolder="${MODELS_DIR}/model_pB_joint_${hash}"
    echo "Fresh run: $modelFolder"
fi

# ---- SIGTERM trap for self-resubmission ----
_resubmit() {
    echo "SIGTERM received — resubmitting with MODEL_FOLDER=$modelFolder"
    sbatch --export=MODEL_FOLDER="$modelFolder" $(realpath "$0")
    exit 0
}
trap '_resubmit' TERM

set +e
train_cae_joint \
    --train-inputs ${trainFile} \
    --test-inputs ${testFile} \
    --model-folder "$modelFolder" \
    --nr-epochs=$TOTAL_EPOCHS \
    --learning-rate=0.0003 \
    --weight-decay=1e-5 \
    --dropout-rate=0.1 \
    --batch-size=512 \
    --architecture=standard \
    --base-channels=64 \
    --output-activation=none \
    --augment \
    --slope-direction-channel=7 \
    --checkpoint-interval=500 \
    --database-path="$databasePath" \
    --lambda-pearson=0.0005 \
    --cn-base-channels=32 \
    --cn-n-conv-layers=4 \
    --cn-cold-threshold-norm=0.3 \
    --lambda-sparsity=0.01 \
    --lambda-cold=0.1 \
    --lambda-mmd=0.1 \
    --lambda-identity=0.1 \
    --max-mask-fraction=0.35 \
    --lambda-mask-cap=1.0 \
    --cold-threshold-k=10.0 \
    --cn-lr=0.0001 \
    $CONTINUE_FLAG
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: train_cae_joint exited with code $TRAIN_EXIT — chain terminated."
    exit $TRAIN_EXIT
fi

EPOCHS_DONE=0
if [ -f "${modelFolder}/history.json" ]; then
    EPOCHS_DONE=$(python3 -c \
        "import json; h=json.load(open('${modelFolder}/history.json')); \
         print(h.get('nr_epochs',0))")
fi

echo "Epochs done: $EPOCHS_DONE / $TOTAL_EPOCHS"

if [ "$EPOCHS_DONE" -ge "$TOTAL_EPOCHS" ]; then
    echo "Training COMPLETE: $modelFolder"
else
    echo "Resubmitting..."
    sbatch --export=MODEL_FOLDER="$modelFolder" $(realpath "$0")
fi

echo "Finished: $(date)"
