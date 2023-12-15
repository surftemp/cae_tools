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

    parser.add_argument("data_path", help="path to netcdf4 file containing data to which model is applied")
    parser.add_argument("output_path", help="path to write the netcdf4 file containing input data plus model outputs")

    parser.add_argument("--model-folder", help="folder to save the trained model to", required=True)
    parser.add_argument("--input-variables", help="name of the input variable(s) in training/test data", required=True)
    parser.add_argument("--prediction-variable", help="name of the prediction variable to create in output data",
                        default="model_output")

    args = parser.parse_args()

    mt = ConvAEModel()
    mt.load(args.model_folder)
    mt.apply(args.data_path, args.input_variables, args.output_path, args.prediction_variable)
