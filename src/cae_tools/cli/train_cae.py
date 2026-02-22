#    Copyright (C) 2023  National Centre for Earth Observation (NCEO)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import json
import os

import xarray as xr
import numpy as np

from cae_tools.models.unet import UNET
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.model_sizer import ModelSpec
from cae_tools.models.preprocessed_dataset import PreprocessedDataset


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-inputs", nargs="+", help="path(s) to netcdf4 file containing training data", required=True)
    parser.add_argument("--test-inputs", nargs="+", help="path(s) to netcdf4 file containing test data", required=True)
    parser.add_argument("--model-folder", help="folder to save the trained model to", required=True)
    parser.add_argument("--continue-training", action="store_true", help="continue training model")
    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=False)
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", required=False)
    parser.add_argument("--preprocessed", action="store_true", help="use preprocessed .pt files instead of netCDF")
    parser.add_argument("--nr-epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--latent-size", type=int, help="size of the latent space", default=4)
    parser.add_argument("--fc-size", type=int, help="size of the fully-connected layers", default=16)
    parser.add_argument("--batch-size", type=int, help="number of images to process in one batch", default=10)
    parser.add_argument("--learning-rate", type=float, help="the learning rate", default=0.001)
    parser.add_argument("--lambda-l1", type=float, help="controls the strength of l1 regularization", default=0.001)
    parser.add_argument("--lambda-pearson", type=float, help="controls the strength of the pearson loss", default=1)
    parser.add_argument("--weight-decay", type=float, help="weight decay coefficient", default=1e-5)
    parser.add_argument("--dropout-rate", type=float, help="dropout rate", default=1e-1)
    parser.add_argument("--method", choices=["unet", "linear"], default="unet", help="model type: unet or linear")
    parser.add_argument("--layer-definitions-path", help="specify path of a JSON file with layer definitions", default=None)
    parser.add_argument("--stride", type=int, help="stride to use in convolutional layers", default=2)
    parser.add_argument("--kernel-size", type=int, help="kernel size to use in convolutional layers", default=3)
    parser.add_argument("--input-layer-count", type=int, help="number of input convolutional layers", default=None)
    parser.add_argument("--output-layer-count", type=int, help="number of output convolutional layers", default=None)
    parser.add_argument("--model-id", type=str, help="specify the model id when creating a model", default=None)
    parser.add_argument("--database-path", type=str, help="path to a database to store evaluation results", default=None)
    parser.add_argument("--chunk-size", type=int, help="chunk size for xarray", default=1000)
    parser.add_argument("--checkpoint-interval", type=int, help="save checkpoint every N epochs (default: no checkpoints)", default=None)
    parser.add_argument("--bottleneck-type", type=str, choices=["fc", "conv"], default="fc",
                        help="bottleneck type: 'fc' for FC bottleneck (default), 'conv' for fully convolutional UNET")
    parser.add_argument("--no-attention", action="store_true", default=False,
                        help="disable channel attention on skip connections")
    parser.add_argument("--skip-mode", type=str, choices=["concat", "add"], default="concat",
                        help="skip connection mode: 'concat' (default) or 'add' (true residual)")
    parser.add_argument("--skip-dropout", type=float, default=0.0,
                        help="probability of dropping entire skip connections during training (default: 0.0)")
    parser.add_argument("--skip-scale", type=float, default=1.0,
                        help="scale factor for skip connections (default: 1.0)")
    parser.add_argument("--latent-activation", type=str, choices=["relu", "leaky_relu", "none"], default="relu",
                        help="activation function for bottleneck FC layers (default: relu)")
    parser.add_argument("--output-activation", type=str, choices=["sigmoid", "tanh", "none"], default="sigmoid",
                        help="output activation: sigmoid for [0,1], tanh for [-1,1], none for unconstrained (default: sigmoid)")
    parser.add_argument("--predict-delta", action="store_true", default=False,
                        help="model predicts delta (LST - ERA5); apply_cae adds ERA5 back for physical LST")
    parser.add_argument("--delta-reference-channel", type=str, default=None,
                        help="input variable name used as reference for delta prediction (e.g. era5_skt)")
    parser.add_argument("--architecture", type=str, choices=["legacy", "standard", "flow_matching"], default="legacy",
                        help="UNet architecture: 'legacy' (original), 'standard' (residual blocks, GroupNorm, bilinear upsample), 'flow_matching' (conditional flow matching)")
    parser.add_argument("--base-channels", type=int, default=64,
                        help="base channel count for standard/flow_matching architecture (default: 64, doubles each stage)")
    parser.add_argument("--flow-steps", type=int, default=4,
                        help="number of Euler integration steps for flow matching inference (default: 4)")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="enable data augmentation (random H/V flips with slope_direction correction)")
    parser.add_argument("--slope-direction-channel", type=int, default=7,
                        help="index of slope_direction channel for augmentation correction (default: 7)")

    args = parser.parse_args()

    # Preprocessed mode - use .pt files
    if args.preprocessed:
        if len(args.train_inputs) != 1 or len(args.test_inputs) != 1:
            raise ValueError("Preprocessed mode requires exactly one .pt file for train and one for test")
        
        print("Loading preprocessed data...")
        train_ds = PreprocessedDataset(args.train_inputs[0])
        test_ds = PreprocessedDataset(args.test_inputs[0])
        
        # Use normalisation parameters from training data for test data
        test_ds.set_normalisation_parameters(train_ds.get_normalisation_parameters())
        
        training_paths = args.train_inputs[0]
        test_paths = args.test_inputs[0]
        
        if args.continue_training:
            parameters_path = os.path.join(args.model_folder, "parameters.json")
            with open(parameters_path) as f:
                parameters = json.loads(f.read())

            if parameters["type"] == "UNET":
                mt = UNET()
            elif parameters["type"] == "LinearModel":
                mt = LinearModel()
            else:
                raise ValueError(f"Unknown model type: {parameters['type']}")

            mt.load(args.model_folder)

            # --nr-epochs is interpreted as the TOTAL epoch target (not additional epochs).
            # This keeps T_max consistent across job boundaries.
            # Compute how many epochs remain for this job.
            already_done = mt.history.get('nr_epochs', 0)
            total_target = args.nr_epochs
            remaining = max(0, total_target - already_done)
            print(f"Continue training: {already_done} epochs done, target {total_target}, running {remaining} this job")
            mt.nr_epochs = remaining
            # Propagate total target so scheduler uses correct T_max
            mt.history['total_nr_epochs'] = total_target

            mt.lr = args.learning_rate
            mt.batch_size = args.batch_size
            mt.checkpoint_interval = args.checkpoint_interval
            mt.lambda_pearson = args.lambda_pearson
        else:
            if args.method == "unet":
                mt = UNET(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                          batch_size=args.batch_size, lr=args.learning_rate, lambda_l1=args.lambda_l1,
                          lambda_pearson=args.lambda_pearson, database_path=args.database_path,
                          weight_decay=args.weight_decay, dropout_rate=args.dropout_rate,
                          checkpoint_interval=args.checkpoint_interval, bottleneck_type=args.bottleneck_type,
                          use_attention=not args.no_attention,
                          skip_mode=args.skip_mode, skip_dropout=args.skip_dropout,
                          skip_scale=args.skip_scale, latent_activation=args.latent_activation,
                          output_activation=args.output_activation,
                          predict_delta=args.predict_delta,
                          delta_reference_channel=args.delta_reference_channel,
                          architecture=args.architecture, base_channels=args.base_channels,
                          flow_steps=args.flow_steps,
                          augment=args.augment, slope_direction_channel=args.slope_direction_channel)
            elif args.method == "linear":
                mt = LinearModel(batch_size=args.batch_size, nr_epochs=args.nr_epochs, lr=args.learning_rate)
            else:
                raise ValueError(f"Unknown method: {args.method}")

            if args.model_id:
                mt.set_model_id(args.model_id)

            if args.layer_definitions_path:
                with open(args.layer_definitions_path) as f:
                    spec = ModelSpec()
                    spec.load(json.loads(f.read()))
                    mt.spec = spec

        mt.train_from_datasets(train_ds, test_ds, model_path=args.model_folder,
                               training_paths=training_paths, testing_paths=test_paths)
        return

    # Original netCDF mode
    if not args.input_variables or not args.output_variable:
        raise ValueError("--input-variables and --output-variable are required for netCDF mode")

    train_ds = [xr.open_dataset(train_input) for train_input in args.train_inputs]
    test_ds = [xr.open_dataset(test_input) for test_input in args.test_inputs]
    case_dimension = train_ds[0][args.output_variable].dims[0]

    train_ds = train_ds[0] if len(train_ds) == 1 else xr.concat(train_ds, dim=case_dimension)
    test_ds = test_ds[0] if len(test_ds) == 1 else xr.concat(test_ds, dim=case_dimension)

    print("Training cases: %d, Test cases: %d" % (train_ds[case_dimension].shape[0], test_ds[case_dimension].shape[0]))

    training_paths = ";".join(args.train_inputs)
    test_paths = ";".join(args.test_inputs)

    # for scalar inputs, broadcast them to have the same dimension as output variable:
    for var in args.input_variables:
        dims = train_ds[var].dims
        if dims == (case_dimension,):
            if 'y' in train_ds.dims and 'x' in train_ds.dims:
                y_dim, x_dim = train_ds.dims['y'], train_ds.dims['x']
            else:
                raise ValueError("'y' and 'x' dimensions not found in the dataset.")

            original_values = train_ds[var].values
            expanded_values = np.broadcast_to(original_values[:, np.newaxis, np.newaxis, np.newaxis],
                                              (original_values.shape[0], 1, y_dim, x_dim))
            expanded_var = xr.DataArray(expanded_values,
                                        coords={case_dimension: train_ds[case_dimension], 'channel': [0],
                                                'y': np.arange(y_dim), 'x': np.arange(x_dim)},
                                        dims=(case_dimension, 'channel', 'y', 'x'))
            train_ds[var] = expanded_var

    for var in args.input_variables:
        dims = test_ds[var].dims
        if dims == (case_dimension,):
            if 'y' in test_ds.dims and 'x' in test_ds.dims:
                y_dim, x_dim = test_ds.dims['y'], test_ds.dims['x']
            else:
                raise ValueError("'y' and 'x' dimensions not found in the dataset.")

            original_values = test_ds[var].values
            expanded_values = np.broadcast_to(original_values[:, np.newaxis, np.newaxis, np.newaxis],
                                              (original_values.shape[0], 1, y_dim, x_dim))
            expanded_var = xr.DataArray(expanded_values,
                                        coords={case_dimension: test_ds[case_dimension], 'channel': [0],
                                                'y': np.arange(y_dim), 'x': np.arange(x_dim)},
                                        dims=(case_dimension, 'channel', 'y', 'x'))
            test_ds[var] = expanded_var

    if args.continue_training:
        parameters_path = os.path.join(args.model_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())

        if parameters["type"] == "UNET":
            mt = UNET()
        elif parameters["type"] == "LinearModel":
            mt = LinearModel()
        else:
            raise ValueError(f"Unknown model type: {parameters['type']}")

        mt.load(args.model_folder)
        # --nr-epochs is the TOTAL epoch target; compute remaining for this job
        already_done = mt.history.get('nr_epochs', 0)
        total_target = args.nr_epochs
        remaining = max(0, total_target - already_done)
        print(f"Continue training: {already_done} epochs done, target {total_target}, running {remaining} this job")
        mt.nr_epochs = remaining
        mt.history['total_nr_epochs'] = total_target
        mt.lr = args.learning_rate
        mt.batch_size = args.batch_size
        mt.checkpoint_interval = args.checkpoint_interval
        mt.lambda_pearson = args.lambda_pearson
    else:
        if args.method == "unet":
            mt = UNET(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                      batch_size=args.batch_size, lr=args.learning_rate, lambda_l1=args.lambda_l1,
                      lambda_pearson=args.lambda_pearson, database_path=args.database_path,
                      weight_decay=args.weight_decay, dropout_rate=args.dropout_rate,
                      checkpoint_interval=args.checkpoint_interval, bottleneck_type=args.bottleneck_type,
                      use_attention=not args.no_attention,
                      skip_mode=args.skip_mode, skip_dropout=args.skip_dropout,
                      skip_scale=args.skip_scale, latent_activation=args.latent_activation,
                      output_activation=args.output_activation,
                      predict_delta=args.predict_delta,
                      delta_reference_channel=args.delta_reference_channel,
                      architecture=args.architecture, base_channels=args.base_channels,
                      flow_steps=args.flow_steps,
                      augment=args.augment, slope_direction_channel=args.slope_direction_channel)
        elif args.method == "linear":
            mt = LinearModel(batch_size=args.batch_size, nr_epochs=args.nr_epochs, lr=args.learning_rate)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        if args.model_id:
            mt.set_model_id(args.model_id)

        # if specified, use the encoder/decoder layer specifications
        if args.layer_definitions_path:
            with open(args.layer_definitions_path) as f:
                spec = ModelSpec()
                spec.load(json.loads(f.read()))
                mt.spec = spec

    mt.train(args.input_variables, args.output_variable, training_ds=train_ds, testing_ds=test_ds,
             model_path=args.model_folder, training_paths=training_paths, testing_paths=test_paths)


if __name__ == '__main__':
    main()
