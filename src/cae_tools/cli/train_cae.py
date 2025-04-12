import argparse
import json
import os
import time

import xarray as xr
import numpy as np
import dask.array as da

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.unet import UNET
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.model_sizer import ModelSpec

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-inputs", nargs="+", help="path(s) to netcdf4 file containing training data", required=True)
    parser.add_argument("--test-inputs", nargs="+", help="path(s) to netcdf4 file containing test data", required=True)
    parser.add_argument("--model-folder", help="folder to save the trained model to", required=True)
    parser.add_argument("--continue-training", action="store_true", help="continue training model")
    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=True)
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", required=True)
    parser.add_argument("--nr-epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--latent-size", type=int, help="size of the latent space", default=4)
    parser.add_argument("--fc-size", type=int, help="size of the fully-connected layers", default=16)
    parser.add_argument("--batch-size", type=int, help="number of images to process in one batch", default=10)
    parser.add_argument("--learning-rate", type=float, help="the learning rate", default=0.001)
    parser.add_argument("--lr-step-size", type=int, help="the schedular steps for the learning rate", default=500)
    parser.add_argument("--lr-gamma", type=float, help="decay factor of the scheduled learning rate", default=0.5)
    parser.add_argument("--lambda-mse", type=float, help="controls the strength of the mse loss in vae", default=1)
    parser.add_argument("--lambda-kl", type=float, help="controls the strength of the kl loss in vae", default=1)
    parser.add_argument("--lambda-l1", type=float, help="controls the strength of l1 regularization", default=0.001)
    parser.add_argument("--lambda-pearson", type=float, help="controls the strength of the pearson loss", default=1)
    parser.add_argument("--lambda-ssim", type=float, help="controls the strength of the ssim loss", default=1)
    parser.add_argument("--lambda-additional", type=float, help="controls the strength of additional regularization", default=1)
    parser.add_argument("--weight-decay", type=float, help="weight decay coefficient", default=1e-5)
    parser.add_argument("--dropout-rate", type=float, help="dropout rate", default=1e-1)
    parser.add_argument("--additional-loss", type=str, help="additional loss types ('contrastive', 'histogram', 'perceptual')", default=None)
    parser.add_argument("--scheduler-type", type=str, help="scheduler type ('StepLR', 'ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR')", default=None)
    parser.add_argument("--method", choices=["conv", "unet", "unet_res", "srcnn_res", "resunet_gan", "var", "vae", "linear"],
                        default="var", help="methods")
    parser.add_argument("--layer-definitions-path", help="specify path of a JSON file with layer definitions", default=None)
    parser.add_argument("--stride", type=int, help="stride to use in convolutional layers", default=2)
    parser.add_argument("--kernel-size", type=int, help="kernel size to use in convolutional layers", default=3)
    parser.add_argument("--input-layer-count", type=int, help="number of input convolutional layers", default=None)
    parser.add_argument("--output-layer-count", type=int, help="number of output convolutional layers", default=None)
    parser.add_argument("--model-id", type=str, help="specify the model id when creating a model", default=None)
    parser.add_argument("--database-path", type=str, help="path to a database to store evaluation results", default=None)
    parser.add_argument("--chunk-size", type=int, help="chunk size for xarray", default=1000)
    parser.add_argument("--include-coasts", help="include coastal areas", default=False)
    parser.add_argument("--mask-variable", type=str, help="name of the mask variable", default=None)

    args = parser.parse_args()
    

    train_ds = xr.open_mfdataset(args.train_inputs,concat_dim="box",combine="nested")
    test_ds = xr.open_mfdataset(args.test_inputs,concat_dim="box",combine="nested")
    case_dimension = train_ds[args.output_variable].dims[0]
    print("Training cases: %d, Test cases: %d"%(train_ds[case_dimension].shape[0],test_ds[case_dimension].shape[0]))


    # Uncomment to filter coastal areas if desired
    # if not args.include_coasts:
    #     train_ds = train_ds.where(train_ds.edge == False, drop=True)
    #     test_ds = test_ds.where(test_ds.edge == False, drop=True)


    training_paths = ";".join(args.train_inputs)
    test_paths = ";".join(args.test_inputs)

    for var in args.input_variables:
        dims = train_ds[var].dims
        if dims == (case_dimension,):
            print(f"Variable '{var}' does not follow the dimension (box, channel, y, x). Extending dimensions...")
            y_dim = train_ds.dims['y']
            x_dim = train_ds.dims['x']

            expanded_var = train_ds[var].expand_dims({'channel': [0]}, axis=1)
            dummy = xr.DataArray(
                da.empty((train_ds[case_dimension].shape[0], 1, y_dim, x_dim), dtype=train_ds[var].dtype),
                dims=(case_dimension, 'channel', 'y', 'x'),
                coords={case_dimension: train_ds[case_dimension],
                        'channel': [0],
                        'y': np.arange(y_dim),
                        'x': np.arange(x_dim)}
            )
            expanded_var = expanded_var.broadcast_like(dummy)
            train_ds[var] = expanded_var

    for var in args.input_variables:
        dims = test_ds[var].dims
        if dims == (case_dimension,):
            print(f"Variable '{var}' does not follow the dimension (box, channel, y, x). Extending dimensions...")
            y_dim = test_ds.dims['y']
            x_dim = test_ds.dims['x']

            expanded_var = test_ds[var].expand_dims({'channel': [0]}, axis=1)
            dummy = xr.DataArray(
                da.empty((test_ds[case_dimension].shape[0], 1, y_dim, x_dim), dtype=test_ds[var].dtype),
                dims=(case_dimension, 'channel', 'y', 'x'),
                coords={case_dimension: test_ds[case_dimension],
                        'channel': [0],
                        'y': np.arange(y_dim),
                        'x': np.arange(x_dim)}
            )
            expanded_var = expanded_var.broadcast_like(dummy)
            test_ds[var] = expanded_var

    if args.continue_training:
        parameters_path = os.path.join(args.model_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())

        if parameters["type"] == "ConvAEModel":
            mt = ConvAEModel()
        elif parameters["type"] == "UNET":
            mt = UNET()
        elif parameters["type"] == "LinearModel":
            mt = LinearModel()
        mt.load(args.model_folder)
        mt.nr_epochs = args.nr_epochs
        mt.lr = args.learning_rate
        mt.batch_size = args.batch_size
    else:
        if args.method == "conv":
            mt = ConvAEModel(fc_size=args.fc_size, encoded_dim_size=args.latent_size,
                             nr_epochs=args.nr_epochs, batch_size=args.batch_size, lr=args.learning_rate)

        elif args.method == "unet":
            mt = UNET(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                      batch_size=args.batch_size, lr=args.learning_rate, lambda_l1=args.lambda_l1,
                      lambda_pearson=args.lambda_pearson, database_path=args.database_path,
                      weight_decay=args.weight_decay, dropout_rate=args.dropout_rate)

        elif args.method == "linear":
            mt = LinearModel(batch_size=args.batch_size, nr_epochs=args.nr_epochs, lr=args.learning_rate)

        if args.model_id:
            mt.set_model_id(args.model_id)

        if args.layer_definitions_path:
            with open(args.layer_definitions_path) as f:
                spec = ModelSpec()
                spec.load(json.loads(f.read()))
                mt.spec = spec


    start_time = time.time()
    print("Ready for training process")
    mt.train(args.input_variables, args.output_variable,
             training_ds=train_ds, testing_ds=test_ds,
             model_path=args.model_folder,
             training_paths=training_paths, testing_paths=test_paths,
             mask_variable_name=args.mask_variable)
    end_time = time.time()
    print(f"Time taken to train: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
