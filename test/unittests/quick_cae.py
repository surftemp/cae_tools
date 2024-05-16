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

import unittest
import os.path
import xarray as xr

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.model_evaluator import ModelEvaluator

import test_specs

data_root_folder = os.path.join(os.path.split(__file__)[0],"..","data")

results_root_folder = os.path.join(os.path.split(__file__)[0],"..","results","cae_quick")

class QuickTest(unittest.TestCase):

    # run selected tests

    def test_tidal(self):
        self.__run("tidal_circle1")

    def test_circle(self):
        self.__run("circle")

    # run all other tests

    def test_others(self):
        for test_spec_name in test_specs.all_specs:
            if test_spec_name not in ["tidal_circle1","circle"]:
                print("Running test:" + test_spec_name)
                self.__run(test_spec_name)

    def __run(self,test_spec_name):
        test_spec = test_specs.all_specs[test_spec_name]
        input_variables = test_spec["inputs"]
        output_variable = test_spec["output"]
        estimated_output_variable = test_spec["output"] + "_estimate"
        (i_h, i_w) = test_spec["input_size"]
        (o_h, o_w) = test_spec["output_size"]
        hyperparameters = test_spec.get("hyperparameters",{})

        folder = os.path.join(data_root_folder, test_spec_name, f"{i_h}x{i_w}_{o_h}x{o_w}")

        if not os.path.exists(folder):
            print("No test data exists for this test.  Run script test/datagen/gen.py to generate the test data first")
            return

        train_path = os.path.join(folder, "train.nc")
        test_path = os.path.join(folder, "test.nc")

        train_ds = xr.open_dataset(train_path)
        test_ds = xr.open_dataset(test_path)

        mt = ConvAEModel(**hyperparameters)
        mt.train(input_variables, output_variable, train_ds, test_ds, training_paths=train_path, test_paths=test_path)
        print(mt.summary())

        results_folder = os.path.join(results_root_folder, test_spec_name, f"{i_h}x{i_w}_{o_h}x{o_w}")
        print("Writing test results to: " + results_folder)
        os.makedirs(results_folder, exist_ok=True)
        model_path = os.path.join(results_folder, "model")
        train_scores_path = os.path.join(results_folder, "train_scores.nc")
        test_scores_path = os.path.join(results_folder, "test_scores.nc")
        mt.save(model_path)

        mt2 = ConvAEModel()
        mt2.load(model_path)

        mt2.apply(train_path, input_variables, train_scores_path, estimated_output_variable)
        mt2.apply(test_path, input_variables, test_scores_path, estimated_output_variable)
        evaluation_html_path = os.path.join(results_folder, "model_evaluation.html")

        me = ModelEvaluator(train_scores_path, test_scores_path, input_variables[0:1], output_variable, evaluation_html_path,
                            estimated_output_variable, model_path)
        me.run()

if __name__ == '__main__':
    unittest.main()

