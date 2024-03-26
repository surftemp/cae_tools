import argparse
import json
import os

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.var_ae_model import VarAEModel
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.model_sizer import ModelSpec

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("training_path",help="path to netcdf4 file containing training data")
    parser.add_argument("test_path", help="path to netcdf4 file containing test data")
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

    args = parser.parse_args()

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
                             batch_size=args.batch_size, lr=args.learning_rate)
        elif args.method == "var":
            mt = VarAEModel(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                        batch_size=args.batch_size, lr=args.learning_rate)
        elif args.method == "linear":
            mt = LinearModel(batch_size=args.batch_size, nr_epochs=args.nr_epochs, lr=args.learning_rate)

        # if specified, use the encoder/decoder layer specifications
        if args.layer_definitions_path:
            with open(args.layer_definitions_path) as f:
                spec = ModelSpec()
                spec.load(json.loads(f.read()))
                mt.spec = spec

    mt.train(args.input_variables, args.output_variable, args.training_path, args.test_path)
    mt.save(args.model_folder)
