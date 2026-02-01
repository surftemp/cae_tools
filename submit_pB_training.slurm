#!/bin/bash
#SBATCH --job-name=pB_unet_train
#SBATCH --exclude=gpuhost004,gpuhost007,gpuhost012
#SBATCH --time=24:00:00
#SBATCH --mem=512G
#SBATCH --nodelist=gpuhost015,gpuhost016
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH -o /home/users/shaerdan/cae_tools_pB/logs/logs_pB_train_%j.out
#SBATCH -e /home/users/shaerdan/cae_tools_pB/logs/logs_pB_train_%j.err

# ============================================================================
# pB Model Training on JASMIN Orchid GPU
# ============================================================================

# Load conda environment
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools   # conda env for cae_tools
export PATH="/gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools/bin:$PATH"
export PYTHONUNBUFFERED=1


# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "=========================================="

# Check GPU is available
nvidia-smi

# Run the training script
./train_pB_model.sh

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
