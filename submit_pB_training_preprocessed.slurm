#!/bin/bash
#SBATCH --job-name=pB_train_prep
#SBATCH --exclude=gpuhost004,gpuhost007,gpuhost012,gpuhost016
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH -o /home/users/shaerdan/cae_tools_pB/logs/logs_pB_train_%j.out
#SBATCH -e /home/users/shaerdan/cae_tools_pB/logs/logs_pB_train_%j.err

# ============================================================================
# pB Model Training using PREPROCESSED data
# Requires: submit_preprocess.slurm to have been run first
# Memory: ~60GB (vs 250GB for netCDF mode)
# ============================================================================

source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "=========================================="

nvidia-smi

./train_pB_model_preprocessed.sh

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
