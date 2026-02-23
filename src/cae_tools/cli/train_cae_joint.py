import argparse
import json
import os
import uuid

from cae_tools.models.unet import UNET
from cae_tools.models.preprocessed_dataset import PreprocessedDataset
from cae_tools.models.unet_joint import JointUNET, load_joint_unet_for_continue


def main():

    parser = argparse.ArgumentParser()

    # ---- Data (always preprocessed .pt — no netCDF mode for joint training) ----
    parser.add_argument("--train-inputs", nargs="+", required=True,
                        help="path to preprocessed training .pt file")
    parser.add_argument("--test-inputs", nargs="+", required=True,
                        help="path to preprocessed test .pt file")
    parser.add_argument("--model-folder", required=True,
                        help="folder to save the trained model to")
    parser.add_argument("--continue-training", action="store_true",
                        help="continue training from model-folder")

    # ---- UNET args — identical to train_cae.py ----
    parser.add_argument("--nr-epochs", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--lambda-pearson", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--latent-size", type=int, default=4)
    parser.add_argument("--fc-size", type=int, default=16)
    parser.add_argument("--bottleneck-type", type=str, choices=["fc", "conv"], default="fc")
    parser.add_argument("--no-attention", action="store_true", default=False)
    parser.add_argument("--skip-mode", type=str, choices=["concat", "add"], default="concat")
    parser.add_argument("--skip-dropout", type=float, default=0.0)
    parser.add_argument("--skip-scale", type=float, default=1.0)
    parser.add_argument("--latent-activation", type=str,
                        choices=["relu", "leaky_relu", "none"], default="relu")
    parser.add_argument("--output-activation", type=str,
                        choices=["sigmoid", "tanh", "none"], default="none")
    parser.add_argument("--predict-delta", action="store_true", default=False)
    parser.add_argument("--delta-reference-channel", type=str, default=None)
    parser.add_argument("--architecture", type=str,
                        choices=["legacy", "standard", "flow_matching"], default="standard")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--slope-direction-channel", type=int, default=7)
    parser.add_argument("--database-path", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--layer-definitions-path", default=None)
    parser.add_argument("--model-id", type=str, default=None)

    # ---- CN-specific args ----
    parser.add_argument("--cn-base-channels", type=int, default=32)
    parser.add_argument("--cn-n-conv-layers", type=int, default=4)
    parser.add_argument("--cn-cold-threshold-norm", type=float, default=0.3)
    parser.add_argument("--lambda-sparsity", type=float, default=0.01)
    parser.add_argument("--lambda-cold", type=float, default=0.1)
    parser.add_argument("--cold-threshold-k", type=float, default=10.0)
    parser.add_argument("--cn-lr", type=float, default=0.0001)
    parser.add_argument("--lambda-mmd", type=float, default=0.1)
    parser.add_argument("--lambda-identity", type=float, default=0.1)
    parser.add_argument("--max-mask-fraction", type=float, default=0.35)
    parser.add_argument("--lambda-mask-cap", type=float, default=1.0)

    args = parser.parse_args()

    # ---- Load datasets (same as train_cae.py preprocessed path) ----
    print("Loading preprocessed data...")
    train_ds = PreprocessedDataset(args.train_inputs[0])
    test_ds = PreprocessedDataset(args.test_inputs[0])
    test_ds.set_normalisation_parameters(train_ds.get_normalisation_parameters())

    training_paths = args.train_inputs[0]
    test_paths = args.test_inputs[0]

    # ---- Build or restore model ----
    if args.continue_training:
        # Load existing joint model — same pattern as train_cae.py continue path
        parameters_path = os.path.join(args.model_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())

        joint = load_joint_unet_for_continue(args.model_folder)

        already_done = joint.unet.history.get('nr_epochs', 0)
        total_target = args.nr_epochs
        remaining = max(0, total_target - already_done)
        print(f"Continue training: {already_done} done, target {total_target}, "
              f"running {remaining} this job")

        # Update mutable training params (same as train_cae.py)
        joint.unet.nr_epochs = remaining
        joint.unet.history['total_nr_epochs'] = total_target
        joint.unet.lr = args.learning_rate
        joint.unet.batch_size = args.batch_size
        joint.unet.checkpoint_interval = args.checkpoint_interval
        joint.unet.lambda_pearson = args.lambda_pearson
        joint.lambda_cold = args.lambda_cold
        joint.lambda_mmd  = args.lambda_mmd
        joint.cn.lambda_identity  = args.lambda_identity
        joint.cn.max_mask_fraction = args.max_mask_fraction
        joint.cn.lambda_mask_cap  = args.lambda_mask_cap

        nr_epochs_this_job = remaining

    else:
        # Fresh run — construct UNET exactly like train_cae.py does
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

        joint = JointUNET(
            unet=mt,
            cn_base_channels=args.cn_base_channels,
            cn_n_conv_layers=args.cn_n_conv_layers,
            cn_cold_threshold_norm=args.cn_cold_threshold_norm,
            lambda_sparsity=args.lambda_sparsity,
            lambda_identity=args.lambda_identity,
            max_mask_fraction=args.max_mask_fraction,
            lambda_mask_cap=args.lambda_mask_cap,
            lambda_cold=args.lambda_cold,
            cold_threshold_k=args.cold_threshold_k,
            cn_lr=args.cn_lr,
            lambda_mmd=args.lambda_mmd,
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
