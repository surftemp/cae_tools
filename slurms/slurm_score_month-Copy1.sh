#!/bin/bash
# =============================================================================
# Submit full scoring pipeline for every date in a given month.
# For each date:
#   1. Array job: score all input boxes (slurm_score_date.sh)
#   2. Collect job: stitch UK map + PNG, delete per-box files (slurm_collect_date.sh)
#   3. Count job: cold box diagnostics on stitched map (slurm_count_cold.sh)
#
# Usage:
#   bash slurm_score_month.sh <YEAR> <MONTH> <CHECKPOINT> <INPUT_BASE> <OUTPUT_BASE> <MODEL_ID>
#
# Example:
#   bash slurm_score_month.sh \
#       2023 08 \
#       /gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_standard_aug_ff1bc219/checkpoint_best_test_mse \
#       /gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs \
#       /gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs \
#       model_pB_standard_aug_ep170
#
# Input directory structure expected:
#   INPUT_BASE/YYYY-MM-DD/input_*.nc
#
# Output structure produced:
#   OUTPUT_BASE/uk_maps/YYYY-MM-DD_<MODEL_ID>.nc
#   OUTPUT_BASE/uk_maps/YYYY-MM-DD_<MODEL_ID>.png
#   Per-box files are deleted after collect step to save disk space.
# =============================================================================

YEAR=$1
MONTH=$2
CHECKPOINT=$3
INPUT_BASE=$4
OUTPUT_BASE=$5
MODEL_ID=$6

if [ -z "$YEAR" ] || [ -z "$MONTH" ] || [ -z "$CHECKPOINT" ] || \
   [ -z "$INPUT_BASE" ] || [ -z "$OUTPUT_BASE" ] || [ -z "$MODEL_ID" ]; then
    echo "Usage: bash slurm_score_month.sh YEAR MONTH CHECKPOINT INPUT_BASE OUTPUT_BASE MODEL_ID"
    exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
MAPS_DIR="${OUTPUT_BASE}/uk_maps"
mkdir -p $MAPS_DIR
mkdir -p /home/users/shaerdan/cae_tools_pB/logs

echo "=========================================="
echo "Scoring month: ${YEAR}-${MONTH}"
echo "Checkpoint:    $CHECKPOINT"
echo "Model ID:      $MODEL_ID"
echo "Input base:    $INPUT_BASE"
echo "Maps output:   $MAPS_DIR"
echo "=========================================="
echo ""

# Iterate over every date in the month that has input data
for INPUT_DIR in ${INPUT_BASE}/${YEAR}-${MONTH}-*/; do
    [ -d "$INPUT_DIR" ] || continue

    DATE=$(basename $INPUT_DIR)

    # Check input files exist
    n_files=$(ls ${INPUT_DIR}/input_*.nc 2>/dev/null | wc -l)
    if [ "$n_files" -eq 0 ]; then
        echo "[$DATE] No input files found, skipping"
        continue
    fi
    N=$((n_files - 1))

    BOX_OUTPUT_DIR="${OUTPUT_BASE}/${DATE}_${MODEL_ID}"
    MAP_OUTPUT="${MAPS_DIR}/${DATE}_${MODEL_ID}.nc"

    # Skip if final map already exists
    if [ -f "$MAP_OUTPUT" ]; then
        echo "[$DATE] Map already exists, skipping: $MAP_OUTPUT"
        continue
    fi

    echo "[$DATE] Submitting pipeline ($n_files files)..."

    # Step 1: Score array job
    SCORE_JOB=$(sbatch --array=0-${N} \
        --parsable \
        ${SCRIPTS_DIR}/slurm_score_date.sh \
        $DATE $CHECKPOINT $INPUT_DIR $BOX_OUTPUT_DIR)
    echo "[$DATE]   Score job:   $SCORE_JOB (array 0-$N)"

    # Step 2: Collect + PNG + cleanup (depends on all score tasks)
    COLLECT_JOB=$(sbatch \
        --parsable \
        --dependency=afterok:${SCORE_JOB} \
        ${SCRIPTS_DIR}/slurm_collect_date.sh \
        $DATE $BOX_OUTPUT_DIR $MAP_OUTPUT $MODEL_ID)
    echo "[$DATE]   Collect job: $COLLECT_JOB"

    # Step 3: Cold box count (depends on collect)
    COUNT_JOB=$(sbatch \
        --parsable \
        --dependency=afterok:${COLLECT_JOB} \
        ${SCRIPTS_DIR}/slurm_count_cold.sh \
        $MAP_OUTPUT "$MODEL_ID $DATE")
    echo "[$DATE]   Count job:   $COUNT_JOB"

    echo ""
done

echo "All jobs submitted."
echo "Monitor with: squeue -u shaerdan"
