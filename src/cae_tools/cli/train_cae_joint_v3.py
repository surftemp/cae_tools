"""
CLI for joint Flow CN v3 + UNET training with learned loss balancing.

Key defaults aligned with standalone train_cae.py:
  --output-activation sigmoid  (was 'none' — caused training divergence)

CN uses soft blending (no threshold). Correction strength is determined
entirely by the flow's learned density via the formula:
  w = sigmoid(-0.5 * z^2)
  lst_cn = w * target + (1 - w) * x_mean

Usage:
  train_cae_joint_v3 --train-inputs ... --test-inputs ... --model-folder ...

Add to setup.py console_scripts:
  train_cae_joint_v3=cae_tools.cli.train_cae_joint_v3:main
"""

import argparse
import json
import os

from cae_tools.models.unet import UNET
from cae_tools.models.preprocessed_dataset import PreprocessedDataset
from cae_tools.models.unet_joint_v3 import JointUNETv3, load_joint_v3_for_continue


def main():
    parser = argparse.ArgumentParser(
        description="Joint Flow CN v3 + UNET training with learned loss balancing"
    )

    # ---- Data ----
    parser.add_argument("--train-inputs", nargs="+", required=True)
    parser.add_argument("--test-inputs", nargs="+", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--continue-training", action="store_true")

    # ---- UNET args ----
    parser.add_argument("--nr-epochs", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--lambda-pearson", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--latent-size", type=int, default=4)
    parser.add_argument("--fc-size", type=int, default=16)
    parser.add_argument("--bottleneck-type", type=str, choices=["fc", "conv"], default="fc")
    parser.add_argument("--no-attention", action="store_true", default=False)
    parser.add_argument("--skip-mode", type=str, choices=["concat", "add"], default="concat")
    parser.add_argument("--skip-dropout", type=float, default=0.0)
    parser.add_argument("--skip-scale", type=float, default=1.0)
    parser.add_argument("--latent-activation", type=str, default="relu")
    parser.add_argument("--output-activation", type=str, default="sigmoid",
                        choices=["sigmoid", "tanh", "none"],
                        help="Output activation (default: sigmoid, matching standalone)")
    parser.add_argument("--predict-delta", action="store_true", default=False)
    parser.add_argument("--delta-reference-channel", type=str, default=None)
    parser.add_argument("--architecture", type=str, default="standard",
                        choices=["legacy", "standard", "flow_matching"])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--slope-direction-channel", type=int, default=7)
    parser.add_argument("--database-path", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--layer-definitions-path", default=None)
    parser.add_argument("--model-id", type=str, default=None)

    # ---- CN v3 flow args ----
    parser.add_argument("--cn-base-channels", type=int, default=64)
    parser.add_argument("--cn-dropout-rate", type=float, default=0.0)
    parser.add_argument("--cn-lr", type=float, default=0.0001)
    parser.add_argument("--cn-pretrain-min-epochs", type=int, default=10)
    parser.add_argument("--cn-pretrain-max-epochs", type=int, default=200)
    parser.add_argument("--cn-convergence-threshold", type=float, default=0.01)
    parser.add_argument("--cn-convergence-window", type=int, default=5)
    parser.add_argument("--lambda-cold", type=float, default=0.1)
    parser.add_argument("--cold-threshold-k", type=float, default=10.0)
    parser.add_argument("--lambda-subgroup", type=float, default=0.1)
    parser.add_argument("--lambda-spectral", type=float, default=0.0,
                        help="Weight for spectral (FFT) loss. 0=disabled.")
    parser.add_argument("--spectral-every-k-epochs", type=int, default=10,
                        help="Compute pattern losses (spectral, and for flow matching "
                        "also Pearson/cold/subgroup) every K-th epoch. "
                        "Default 10.")
    parser.add_argument("--flow-n-coupling-layers", type=int, default=4)
    parser.add_argument("--flow-coupling-hidden", type=int, default=64)

    args = parser.parse_args()

    # ---- Load datasets ----
    print("Loading preprocessed data...")
    train_ds = PreprocessedDataset(args.train_inputs[0])
    test_ds = PreprocessedDataset(args.test_inputs[0])
    test_ds.set_normalisation_parameters(train_ds.get_normalisation_parameters())

    training_paths = args.train_inputs[0]
    test_paths = args.test_inputs[0]

    # ---- Build or restore model ----
    if args.continue_training:
        joint = load_joint_v3_for_continue(args.model_folder)

        already_done = joint.unet.history.get('nr_epochs', 0)
        total_target = args.nr_epochs
        remaining = max(0, total_target - already_done)
        print(f"Continue: {already_done} done, target {total_target}, "
              f"running {remaining}")

        joint.unet.nr_epochs = remaining
        joint.unet.history['total_nr_epochs'] = total_target
        joint.unet.lr = args.learning_rate
        joint.unet.batch_size = args.batch_size
        joint.unet.checkpoint_interval = args.checkpoint_interval
        joint.unet.lambda_pearson = args.lambda_pearson
        joint.lambda_cold = args.lambda_cold
        joint.lambda_subgroup = args.lambda_subgroup
        joint.lambda_spectral = args.lambda_spectral
        joint.spectral_every_k_epochs = args.spectral_every_k_epochs

        nr_epochs_this_job = remaining
    else:
        from cae_tools.models.model_sizer import ModelSpec

        mt = UNET(
            fc_size=args.fc_size,
            encoded_dim_size=args.latent_size,
            nr_epochs=args.nr_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            lambda_pearson=args.lambda_pearson,
            database_path=args.database_path,
            weight_decay=args.weight_decay,
            dropout_rate=args.dropout_rate,
            checkpoint_interval=args.checkpoint_interval,
            bottleneck_type=args.bottleneck_type,
            use_attention=not args.no_attention,
            skip_mode=args.skip_mode,
            skip_dropout=args.skip_dropout,
            skip_scale=args.skip_scale,
            latent_activation=args.latent_activation,
            output_activation=args.output_activation,
            predict_delta=args.predict_delta,
            delta_reference_channel=args.delta_reference_channel,
            architecture=args.architecture,
            base_channels=args.base_channels,
            flow_steps=args.flow_steps,
            augment=args.augment,
            slope_direction_channel=args.slope_direction_channel,
        )

        if args.model_id:
            mt.set_model_id(args.model_id)

        if args.layer_definitions_path:
            with open(args.layer_definitions_path) as f:
                spec = ModelSpec()
                spec.load(json.loads(f.read()))
                mt.spec = spec

        joint = JointUNETv3(
            unet=mt,
            cn_base_channels=args.cn_base_channels,
            cn_dropout_rate=args.cn_dropout_rate,
            cn_lr=args.cn_lr,
            cn_pretrain_min_epochs=args.cn_pretrain_min_epochs,
            cn_pretrain_max_epochs=args.cn_pretrain_max_epochs,
            cn_convergence_threshold=args.cn_convergence_threshold,
            cn_convergence_window=args.cn_convergence_window,
            lambda_cold=args.lambda_cold,
            cold_threshold_k=args.cold_threshold_k,
            lambda_subgroup=args.lambda_subgroup,
            lambda_spectral=args.lambda_spectral,
            spectral_every_k_epochs=args.spectral_every_k_epochs,
            flow_n_coupling_layers=args.flow_n_coupling_layers,
            flow_coupling_hidden=args.flow_coupling_hidden,
        )

        nr_epochs_this_job = args.nr_epochs

    # ---- Train ----
    joint.train_joint(
        train_ds=train_ds,
        test_ds=test_ds,
        model_folder=args.model_folder,
        nr_epochs=nr_epochs_this_job,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        database_path=args.database_path,
        training_paths=training_paths,
        test_paths=test_paths,
    )


if __name__ == '__main__':
    main()
