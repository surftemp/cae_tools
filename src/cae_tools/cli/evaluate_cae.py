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

from cae_tools.models.model_evaluator import ModelEvaluator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-inputs", nargs="+", help="path to netcdf4 file(s) containing training data")
    parser.add_argument("--test-inputs", nargs="+", help="path to netcdf4 file(s) containing test data")

    parser.add_argument("--output-html-folder", help="folder to write output html to",default="")
    parser.add_argument("--input-variables", nargs="*", help="input variables to plot")
    parser.add_argument("--sample-count", type=int, help="fraction of cases to plot for each partition", default=None)
    parser.add_argument("--model-folder", help="folder to save the trained model to", required=True)
    parser.add_argument("--prediction-variable", help="name of the prediction variable to create in output data",
                       default=None)
    parser.add_argument("--x-coordinate", help="name of the x-coordinate", default=None)
    parser.add_argument("--y-coordinate", help="name of the y-coordinate", default=None)
    parser.add_argument("--time-coordinate", help="name of the time-coordinate", default=None)


    parser.add_argument("--database-path", type=str, help="path to a database to store evaluation results", default=None)

    args = parser.parse_args()

    mt = ModelEvaluator(training_paths=args.train_inputs,
                        testing_paths=args.test_inputs,
                        output_html_folder=args.output_html_folder,
                        model_path=args.model_folder,
                        model_output_variable=args.prediction_variable,
                        input_variables=args.input_variables,
                        sample_count=args.sample_count,
                        database_path=args.database_path,
                        x_coordinate=args.x_coordinate,
                        y_coordinate=args.y_coordinate,
                        time_coordinate=args.time_coordinate)

    mt.run()
