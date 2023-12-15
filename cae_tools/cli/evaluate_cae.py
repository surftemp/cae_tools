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

from cae_tools.utils.evaluate import ModelEvaluator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("training_path",help="path to netcdf4 file containing training data")
    parser.add_argument("test_path", help="path to netcdf4 file containing test data")
    parser.add_argument("output_html_path", help="path to write output html to")

    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=True)
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", required=True)

    parser.add_argument("--model-folder", help="folder to save the trained model to", default="")
    parser.add_argument("--prediction-variable", help="name of the prediction variable to create in output data",
                       default="")

    args = parser.parse_args()

    mt = ModelEvaluator(train_path=args.training_path,
                        test_path=args.test_path,
                        input_variables=args.input_variables,
                        output_variable=args.output_variable,
                        output_html_path=args.output_html_path,
                        model_path=args.model_folder,
                        model_output_variable=args.prediction_variable)
    mt.run()
