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
    parser.add_argument("--model_folder", help="path to save the trained model to",default="")
    parser.add_argument("--input-variable", help="name of the input variable in training/test data", default="input")
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", default="output")
    parser.add_argument("--nr-epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--latent-size", type=int, help="size of the latent space", default=4)
    parser.add_argument("--fc-size", type=int, help="size of the fully-connected layers", default=16)

    args = parser.parse_args()

    mt = ConvAEModel(args.input_variable, args.output_variable, fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs)
    mt.train(args.training_path, args.test_path)
    if args.model_folder:
        mt.save(args.model_folder)
