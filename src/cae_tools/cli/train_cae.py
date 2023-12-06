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

from cae_tools.models.conv_ae_model import ConvAEModel


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("training_path",help="path to netcdf4 file containing training data")
    parser.add_argument("test_path", help="path to netcdf4 file containing test data")
    parser.add_argument("--model-folder", help="folder to save the trained model to",required=True)
    parser.add_argument("--continue-training", action="store_true", help="continue training model")
    parser.add_argument("--input-variable", help="name of the input variable in training/test data", default="input")
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", default="output")
    parser.add_argument("--nr-epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--latent-size", type=int, help="size of the latent space", default=4)
    parser.add_argument("--fc-size", type=int, help="size of the fully-connected layers", default=16)
    parser.add_argument("--batch-size", type=int, help="number of images to process in one batch", default=10)
    parser.add_argument("--learning-rate", type=float, help="controls the rate at which model weights are updated", default=0.001)

    args = parser.parse_args()

    if args.continue_training:
        mt = ConvAEModel()
        mt.load(args.model_folder)
    else:
        mt = ConvAEModel(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                     batch_size=args.batch_size, lr=args.learning_rate)
    mt.train(args.input_variable, args.output_variable, args.training_path, args.test_path)
    mt.save(args.model_folder)

