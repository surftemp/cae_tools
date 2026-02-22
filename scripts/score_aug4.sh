#!/bin/bash
#SBATCH --job-name=score_aug4
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=eocis_chuk
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-45
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/score_aug4_%A_%a.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/score_aug4_%A_%a.err

source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"

# CHECKPOINT=/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_standard_aug_ff1bc219/checkpoint_best_test_mse
# INPUT_DIR=/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04
# OUTPUT_DIR=/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04_standard_aug_ep170

CHECKPOINT=/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_standard_aug_k6JJ0fYm/checkpoint_epoch_500
OUTPUT_DIR=/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04_standard_aug_k6JJ_ep500
INPUT_DIR=/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04



mkdir -p $OUTPUT_DIR

# Pick the file for this array task
files=($INPUT_DIR/input_*.nc)
f=${files[$SLURM_ARRAY_TASK_ID]}
fname=$(basename $f)

echo "Task $SLURM_ARRAY_TASK_ID: $fname"
python /home/users/shaerdan/cae_tools_pB/src/cae_tools/cli/apply_cae.py \
    $f $OUTPUT_DIR/$fname \
    --model-folder $CHECKPOINT