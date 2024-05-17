import argparse
import json
import os

import xarray as xr

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.var_ae_model import VarAEModel
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.model_sizer import ModelSpec

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-inputs", nargs="+", help="path(s) to netcdf4 file containing training data", required=True)
    parser.add_argument("--test-inputs", nargs="+", help="path(s) to netcdf4 file containing test data", required=True)
    parser.add_argument("--model-folder", help="folder to save the trained model to",required=True)
    parser.add_argument("--continue-training", action="store_true", help="continue training model")
    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=True)
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", required=True)
    parser.add_argument("--nr-epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--latent-size", type=int, help="size of the latent space", default=4)
    parser.add_argument("--fc-size", type=int, help="size of the fully-connected layers", default=16)
    parser.add_argument("--batch-size", type=int, help="number of images to process in one batch", default=10)
    parser.add_argument("--learning-rate", type=float, help="controls the rate at which model weights are updated", default=0.001)
    parser.add_argument("--method", choices=["conv", "var", "linear"], default="var", help="model training method: 'conv' for ConvAEModel or 'var' for VarAEModel or 'linear' for LinearModel")
    parser.add_argument("--layer-definitions-path", help="specify path of a JSON file with layer definitions", default=None)
    parser.add_argument("--stride", type=int, help="stride to use in convolutional layers", default=2)
    parser.add_argument("--kernel-size", type=int, help="kernel size to use in convolutional layers", default=3)
    parser.add_argument("--input-layer-count", type=int, help="number of input convolutional layers", default=None)
    parser.add_argument("--output-layer-count", type=int, help="number of output convolutional layers", default=None)
    parser.add_argument("--model-id", type=str, help="specify the model id when creating a model", default=None)
    parser.add_argument("--database-path", type=str, help="path to a database to store evaluation results",
                        default=None)

    args = parser.parse_args()

    train_ds = [xr.open_dataset(train_input) for train_input in args.train_inputs]
    test_ds = [xr.open_dataset(test_input) for test_input in args.test_inputs]
    target_dimension = train_ds[0][args.output_variable].dims[0]


    train_ds = train_ds[0] if len(train_ds) == 1 else xr.concat(train_ds, dim=target_dimension)
    test_ds = test_ds[0] if len(test_ds) == 1 else xr.concat(test_ds, dim=target_dimension)

    print("Training cases: %d, Test cases: %d"%(train_ds[target_dimension].shape[0],test_ds[target_dimension].shape[0]))

    training_paths = ";".join(args.train_inputs)
    test_paths = ";".join(args.test_inputs)

    if args.continue_training:
        parameters_path = os.path.join(args.model_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())

        if parameters["type"] == "ConvAEModel":
            mt = ConvAEModel()
        elif parameters["type"] == "VarAEModel":
            mt = VarAEModel()
        elif parameters["type"] == "LinearModel":
            mt = LinearModel()
        mt.load(args.model_folder)
        # update selected parameters from the command line args
        mt.nr_epochs = args.nr_epochs
        mt.lr = args.learning_rate
        mt.batch_size = args.batch_size
    else:
        if args.method == "conv":
            mt = ConvAEModel(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                             batch_size=args.batch_size, lr=args.learning_rate, database_path=args.database_path)
        elif args.method == "var":
            mt = VarAEModel(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                        batch_size=args.batch_size, lr=args.learning_rate)
        elif args.method == "linear":
            mt = LinearModel(batch_size=args.batch_size, nr_epochs=args.nr_epochs, lr=args.learning_rate)

        if args.model_id:
            mt.set_model_id(args.model_id)

        # if specified, use the encoder/decoder layer specifications
        if args.layer_definitions_path:
            with open(args.layer_definitions_path) as f:
                spec = ModelSpec()
                spec.load(json.loads(f.read()))
                mt.spec = spec

    mt.train(args.input_variables, args.output_variable, training_ds=train_ds, testing_ds=test_ds, model_path=args.model_folder, training_paths=training_paths,
             testing_paths=test_paths)

if __name__ == '__main__':
    main()
