#!/bin/bash
#SBATCH --job-name=score_date
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=eocis_chuk
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH --output=/home/users/shaerdan/cae_tools_pB/logs/score_date_%A_%a.out
#SBATCH --error=/home/users/shaerdan/cae_tools_pB/logs/score_date_%A_%a.err
# =============================================================================
# Score one date: one SLURM array task per input .nc file
#
# Usage:
#   sbatch --array=0-N slurm_score_date.sh <DATE> <CHECKPOINT> <INPUT_DIR> <OUTPUT_DIR> [--overwrite]
#
# Flags:
#   --overwrite   Re-score even if output box file already exists
#
# Default behaviour: skip if output already exists
# =============================================================================

DATE=$1
CHECKPOINT=$2
INPUT_DIR=$3
OUTPUT_DIR=$4
OVERWRITE=${5:-""}

if [ -z "$DATE" ] || [ -z "$CHECKPOINT" ] || [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: missing arguments"
    echo "Usage: sbatch --array=0-N slurm_score_date.sh DATE CHECKPOINT INPUT_DIR OUTPUT_DIR [--overwrite]"
    exit 1
fi

set --
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"

mkdir -p $OUTPUT_DIR

# Pick the file for this array task
files=($INPUT_DIR/input_*.nc)
f=${files[$SLURM_ARRAY_TASK_ID]}
fname=$(basename $f)
out_path=$OUTPUT_DIR/$fname

echo "Task $SLURM_ARRAY_TASK_ID: $fname"
echo "Checkpoint: $CHECKPOINT"

if [ -f "$out_path" ] && [ "$OVERWRITE" != "--overwrite" ]; then
    echo "Output already exists, skipping: $out_path"
    exit 0
fi

python /home/users/shaerdan/cae_tools_pB/src/cae_tools/cli/apply_cae.py \
    $f $out_path \
    --model-folder $CHECKPOINT

echo "Done: $fname"
