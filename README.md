# cae_tools Updates: Preprocessed Data Support + GPU Memory Fix

## Changes Summary

### New Files:
1. `preprocess_data.py` - CLI script to convert netCDF files to efficient .pt format
2. `preprocessed_dataset.py` - PyTorch Dataset class for loading preprocessed data
3. `submit_preprocess.slurm` - SLURM script to run preprocessing
4. `train_pB_model_preprocessed.sh` - Training script using preprocessed data
5. `submit_pB_training_preprocessed.slurm` - SLURM script for training

### Modified Files:
6. `train_cae.py` - Added `--preprocessed` flag to use .pt files
7. `unet.py` - Added `train_from_datasets()` method + **GPU memory fix**
8. `setup.cfg` - Added `preprocess_data` entry point

## Installation

**Create a new branch first:**

```bash
cd ~/cae_tools_pB
git checkout -b preprocessed_data_support
```

**Copy files:**

```bash
# New CLI script
cp preprocess_data.py src/cae_tools/cli/

# New dataset class
cp preprocessed_dataset.py src/cae_tools/models/

# Modified files
cp train_cae.py src/cae_tools/cli/
cp unet.py src/cae_tools/models/
cp setup.cfg .

# SLURM and shell scripts
cp submit_preprocess.slurm .
cp train_pB_model_preprocessed.sh .
cp submit_pB_training_preprocessed.slurm .
chmod +x train_pB_model_preprocessed.sh

# Re-install to register preprocess_data command
pip install -e .
```

## Usage Workflow

### Step 1: Preprocess Data (run ONCE)

```bash
sbatch submit_preprocess.slurm
```

This creates:
- `/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/train_v8.pt`
- `/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/test_v8.pt`

Wait for this to complete before training.

### Step 2: Train Model (run as many times as needed)

```bash
sbatch submit_pB_training_preprocessed.slurm
```

## Memory Requirements

| Mode | Preprocessing | Training |
|------|---------------|----------|
| **Preprocessed** | 96 GB RAM | 96 GB RAM |
| **Original netCDF** | N/A | 250+ GB RAM |

## GPU Memory Fix

Both training modes now keep batches on CPU and move to GPU per-batch.
This fixes VRAM OOM on large datasets (adds ~5 sec overhead per epoch).

## Reverting Changes

```bash
cd ~/cae_tools_pB
git checkout pB_stable
```

## Inference

No changes needed! Model files are saved in the same format.
`apply_cae` continues to work with netCDF files as before.
