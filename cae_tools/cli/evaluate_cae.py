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
    parser.add_argument("--layer-config-path", help="JSON file specifying one or more layer specifications", default="")
    parser.add_argument("--output-html-folder", help="folder to write output html to",default="")
    parser.add_argument("--time-variable", help="name of a variable containing the acquisition times of each case", default="")
    parser.add_argument("--coordinates", nargs=2, type=str, metavar=("X-VARIABLE","Y-VARIABLE"), required=False)
    parser.add_argument("--with-osm", action="store_true", help="add an open streetmap layer")
    parser.add_argument("--case-sample-fraction", type=float, help="output to html details on this fraction of cases", required=False)
    parser.add_argument("--model-folder", help="folder to save the trained model to", required=True)
    parser.add_argument("--prediction-variable", help="name of the prediction variable to create in output data", default=None)

    parser.add_argument("--database-path", type=str, help="path to a database to store evaluation results", default=None)

    args = parser.parse_args()

    mt = ModelEvaluator(training_paths=args.train_inputs,
                        testing_paths=args.test_inputs,
                        layer_config_path=args.layer_config_path,
                        output_html_folder=args.output_html_folder,
                        model_path=args.model_folder,
                        model_output_variable=args.prediction_variable,
                        time_variable=args.time_variable,
                        database_path=args.database_path,
                        with_osm=args.with_osm,
                        coordinates=args.coordinates,
                        case_sample_fraction=args.case_sample_fraction)
    mt.run()
