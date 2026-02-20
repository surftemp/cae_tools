#!/bin/bash
#SBATCH --job-name=pB_add_sd_nl
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --chdir=/home/users/shaerdan/cae_tools_pB
#SBATCH -o /home/users/shaerdan/cae_tools_pB/logs/pB_add_skipdrop_nolrelu_%j.out
#SBATCH -e /home/users/shaerdan/cae_tools_pB/logs/pB_add_skipdrop_nolrelu_%j.err


# Load conda environment
source ~/miniforge3/bin/activate
conda activate /gws/nopw/j04/eocis_chuk/shaerdan/envs/pyt_cae_tools
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

cd /home/users/shaerdan/cae_tools_pB/

bash train_pB_add_skipdrop_nolrelu.sh
