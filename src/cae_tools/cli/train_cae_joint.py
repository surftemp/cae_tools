"""
CLI entry point for joint CN + UNET training.

Usage:
    train_cae_joint \
        --train-inputs /path/to/train_v8_zscore_rm_coldpattern.pt \
        --test-inputs  /path/to/test_v8_zscore_rm_coldpattern.pt \
        --model-folder /path/to/model_pB_joint_HASH \
        --nr-epochs 3500 \
        --batch-size 512 \
        --architecture standard \
        --base-channels 64 \
        --output-activation none \
        --augment \
        --slope-direction-channel 7 \
        --database-path database_pB_joint.db \
        --checkpoint-interval 500 \
        --cn-base-channels 32 \
        --cn-n-conv-layers 4 \
        --lambda-cold 0.1 \
        --lambda-sparsity 0.01 \
        --cn-lr 0.0001

Continue training:
    MODEL_FOLDER=/path/to/existing/model_pB_joint_HASH train_cae_joint \
        --train-inputs ... --test-inputs ... [same args]
        --continue-training
"""

import argparse
import json
import os
import sys
import uuid

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description='Joint CN + UNET training for LST downscaling'
    )

    # ---- Data ----
    p.add_argument('--train-inputs', nargs='+', required=True,
                   help='Path(s) to preprocessed training .pt file(s)')
    p.add_argument('--test-inputs', nargs='+', required=True,
                   help='Path(s) to preprocessed test .pt file(s)')

    # ---- Model folder ----
    p.add_argument('--model-folder', type=str, default=None,
                   help='Output model folder. Auto-generated if not specified.')
    p.add_argument('--continue-training', action='store_true',
                   help='Continue training from MODEL_FOLDER (or env var MODEL_FOLDER)')
    p.add_argument('--models-dir',
                   default='/gws/nopw/j04/eocis_chuk/shaerdan/models',
                   help='Root directory for model folders when auto-generating name')

    # ---- UNET architecture (mirrors train_cae.py) ----
    p.add_argument('--architecture', default='standard',
                   choices=['legacy', 'standard', 'flow_matching'])
    p.add_argument('--base-channels', type=int, default=64)
    p.add_argument('--output-activation', default='none',
                   choices=['none', 'sigmoid', 'tanh'])
    p.add_argument('--no-attention', action='store_true')
    p.add_argument('--skip-mode', default='concat', choices=['concat', 'add'])
    p.add_argument('--latent-activation', default='relu',
                   choices=['relu', 'leaky_relu', 'none'])
    p.add_argument('--augment', action='store_true')
    p.add_argument('--slope-direction-channel', type=int, default=7)
    p.add_argument('--predict-delta', action='store_true')
    p.add_argument('--delta-reference-channel', type=int, default=None)

    # ---- UNET training hyperparams ----
    p.add_argument('--nr-epochs', type=int, default=3500)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=0.0003)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--dropout-rate', type=float, default=0.1)
    p.add_argument('--lambda-pearson', type=float, default=0.0,
                   help='Pearson loss weight (default 0 for joint training)')
    p.add_argument('--checkpoint-interval', type=int, default=500)
    p.add_argument('--database-path', type=str, default=None)
    p.add_argument('--test-interval', type=int, default=10)

    # ---- CN hyperparams ----
    p.add_argument('--cn-base-channels', type=int, default=32,
                   help='CN CNN width')
    p.add_argument('--cn-n-conv-layers', type=int, default=4,
                   help='CN CNN depth')
    p.add_argument('--cn-cold-threshold-norm', type=float, default=0.3,
                   help='Normalised delta prior threshold for CN mask initialisation')
    p.add_argument('--lambda-sparsity', type=float, default=0.01,
                   help='CN L1 sparsity regularisation weight')
    p.add_argument('--lambda-cold', type=float, default=0.1,
                   help='Cold pixel rate loss weight')
    p.add_argument('--cold-threshold-k', type=float, default=10.0,
                   help='K — threshold for soft cold pixel rate detection')
    p.add_argument('--cn-lr', type=float, default=0.0001,
                   help='CN learning rate (typically lower than UNET lr)')

    return p.parse_args()


def main():
    args = parse_args()

    # ---- Imports ----
    from cae_tools.models.preprocessed_dataset import PreprocessedDataset
    from cae_tools.models.unet import UNET
    from cae_tools.models.unet_joint import (
        JointUNET,
        build_joint_unet_fresh,
        load_joint_unet_for_continue,
    )

    # ---- Resolve model folder ----
    model_folder = args.model_folder
    continue_training = args.continue_training

    # Also check environment variable (for SLURM self-resubmission)
    env_model_folder = os.environ.get('MODEL_FOLDER', '')
    if env_model_folder and os.path.exists(
            os.path.join(env_model_folder, 'parameters.json')):
        model_folder = env_model_folder
        continue_training = True
        print(f"Continuing from MODEL_FOLDER env var: {model_folder}")

    if continue_training and model_folder:
        print(f"Continue training from: {model_folder}")
        joint = load_joint_unet_for_continue(model_folder)
    else:
        # Fresh run
        hash_id = uuid.uuid4().hex[:8]
        if model_folder is None:
            model_folder = os.path.join(
                args.models_dir, f'model_pB_joint_{hash_id}'
            )
        print(f"Fresh joint training run: {model_folder}")

        # Build UNET
        unet_kwargs = dict(
            nr_epochs=args.nr_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout_rate=args.dropout_rate,
            lambda_pearson=args.lambda_pearson,
            architecture=args.architecture,
            base_channels=args.base_channels,
            output_activation=args.output_activation,
            use_attention=not args.no_attention,
            skip_mode=args.skip_mode,
            latent_activation=args.latent_activation,
            augment=args.augment,
            slope_direction_channel=args.slope_direction_channel,
            predict_delta=args.predict_delta,
            delta_reference_channel=args.delta_reference_channel,
            database_path=args.database_path,
            test_interval=args.test_interval,
            checkpoint_interval=args.checkpoint_interval,
        )

        joint_kwargs = dict(
            cn_base_channels=args.cn_base_channels,
            cn_n_conv_layers=args.cn_n_conv_layers,
            cn_cold_threshold_norm=args.cn_cold_threshold_norm,
            lambda_sparsity=args.lambda_sparsity,
            lambda_cold=args.lambda_cold,
            cold_threshold_k=args.cold_threshold_k,
            cn_lr=args.cn_lr,
        )

        joint = JointUNET(UNET(**unet_kwargs), **joint_kwargs)

    # ---- Load datasets ----
    print("Loading training dataset...")
    train_ds = PreprocessedDataset(args.train_inputs[0])

    print("Loading test dataset...")
    test_ds = PreprocessedDataset(args.test_inputs[0])

    # Set normalisation from training data
    norm_params = train_ds.get_normalisation_parameters()
    test_ds.set_normalisation_parameters(norm_params)

    # Initialise UNET input/output shapes from dataset if fresh run
    if not continue_training:
        sample = train_ds[0]
        sample_input  = sample[0]
        sample_output = sample[1]
        joint.unet.input_shape = tuple(sample_input.shape)
        joint.unet.output_shape = tuple(sample_output.shape)
        joint.unet.normalisation_parameters = norm_params
        joint.unet._init_architecture()

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    print(f"Input shape:   {joint.unet.input_shape}")
    print(f"Output shape:  {joint.unet.output_shape}")

    # ---- Run joint training ----
    joint.train_joint(
        train_dataset=train_ds,
        test_dataset=test_ds,
        model_folder=model_folder,
        nr_epochs=args.nr_epochs,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        database_path=args.database_path,
    )


if __name__ == '__main__':
    main()
