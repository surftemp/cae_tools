#!/bin/bash
#SBATCH --job-name=pB_residual
#SBATCH -o pB_residual.out
#SBATCH -e pB_residual.err
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=standard

conda activate pyt_cae_tools
bash train_pB_residual.sh
