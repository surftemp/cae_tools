#!/bin/bash
# =============================================================================
# Launch all 4 experiments on orchid
# Run from /home/users/shaerdan/cae_tools_pB
#
# Prerequisites:
#   - Updated unet.py and train_cae.py installed in pyt_cae_tools env
#   - pB_spec_add.json placed in /home/users/shaerdan/cae_tools_pB/
#   - logs/ directory exists
#
# Experiments:
#   A: skip_mode=add                              (tests decoder suppression)
#   B: skip_dropout=0.3                           (forces bottleneck usage)
#   C: skip_mode=add + latent_activation=none     (add + fix latent sparsity)
#   D: skip_mode=add + skip_dropout=0.3 + latent_activation=none (full combo)
# =============================================================================

mkdir -p logs

echo "Submitting Experiment A: Additive skip connections"
sbatch submit_pB_add.slurm

echo "Submitting Experiment B: Skip dropout"
sbatch submit_pB_skipdrop.slurm

echo "Submitting Experiment C: Additive skips + no latent ReLU"
sbatch submit_pB_add_nolrelu.slurm

echo "Submitting Experiment D: Full combination"
sbatch submit_pB_add_skipdrop_nolrelu.slurm

echo ""
echo "All experiments submitted. Check queue with: squeue -u shaerdan"
