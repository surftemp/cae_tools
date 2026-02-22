#!/bin/bash
# =============================================================================
# Submit full scoring pipeline for every date in a given month.
# For each date:
#   1. Array job: score all input boxes (slurm_score_date.sh)
#   2. Collect job: stitch UK map + PNG, delete per-box files (slurm_collect_date.sh)
#   3. Count job: cold pixel diagnostics - ALWAYS runs (slurm_count_cold.sh)
#
# Usage:
#   bash slurm_score_month.sh <YEAR> <MONTH> <CHECKPOINT> <INPUT_BASE> <OUTPUT_BASE> <MODEL_ID> [--overwrite]
#
# Flags:
#   --overwrite   Pass through to score and collect steps - reruns everything
#
# Default behaviour:
#   - Score tasks skip if per-box output already exists
#   - Collect skips if UK map already exists
#   - Count cold ALWAYS runs regardless
#
# The only reason the full chain is skipped for a date is if INPUT_DIR does
# not exist (no data to process).
# =============================================================================

YEAR=$1
MONTH=$2
CHECKPOINT=$3
INPUT_BASE=$4
OUTPUT_BASE=$5
MODEL_ID=$6
OVERWRITE=${7:-""}

if [ -z "$YEAR" ] || [ -z "$MONTH" ] || [ -z "$CHECKPOINT" ] || \
   [ -z "$INPUT_BASE" ] || [ -z "$OUTPUT_BASE" ] || [ -z "$MODEL_ID" ]; then
    echo "Usage: bash slurm_score_month.sh YEAR MONTH CHECKPOINT INPUT_BASE OUTPUT_BASE MODEL_ID [--overwrite]"
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
echo "Overwrite:     ${OVERWRITE:-no}"
echo "=========================================="
echo ""

for INPUT_DIR in ${INPUT_BASE}/${YEAR}-${MONTH}-*/; do
    [ -d "$INPUT_DIR" ] || continue

    DATE=$(basename $INPUT_DIR)

    # Only legitimate reason to skip entire chain: no input data exists
    n_files=$(ls ${INPUT_DIR}/input_*.nc 2>/dev/null | wc -l)
    if [ "$n_files" -eq 0 ]; then
        echo "[$DATE] No input files found, skipping"
        continue
    fi
    N=$((n_files - 1))

    BOX_OUTPUT_DIR="${OUTPUT_BASE}/${DATE}_${MODEL_ID}"
    MAP_OUTPUT="${MAPS_DIR}/${DATE}_${MODEL_ID}.nc"

    echo "[$DATE] Submitting pipeline ($n_files files)..."

    # Step 1: Score array job (has per-file skip logic internally)
    SCORE_JOB=$(sbatch --array=0-${N} \
        --parsable \
        ${SCRIPTS_DIR}/slurm_score_date.sh \
        $DATE $CHECKPOINT $INPUT_DIR $BOX_OUTPUT_DIR $OVERWRITE)
    echo "[$DATE]   Score job:   $SCORE_JOB (array 0-$N)"

    # Step 2: Collect + PNG + cleanup (has map-exists skip logic internally)
    COLLECT_JOB=$(sbatch \
        --parsable \
        --dependency=afterok:${SCORE_JOB} \
        ${SCRIPTS_DIR}/slurm_collect_date.sh \
        $DATE $BOX_OUTPUT_DIR $MAP_OUTPUT $MODEL_ID $OVERWRITE)
    echo "[$DATE]   Collect job: $COLLECT_JOB"

    # Step 3: Count cold - always runs, no skip logic
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
