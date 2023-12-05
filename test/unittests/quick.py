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

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.utils.evaluate import ModelEvaluator

import test_specs

data_root_folder = os.path.join(os.path.split(__file__)[0],"..","data")


results_root_folder = os.path.join(os.path.split(__file__)[0],"..","results")

class QuickTest(unittest.TestCase):

    def test_all(self):
        for test_spec in test_specs.all_specs:
            print("Running test:"+str(test_spec))
            self.__run(test_spec)

    def test_tidal(self):
        self.__run(((6,6),(256,256),"tidal_circle"))

    def __run(self,test_spec):
        input_variable = "lowres"
        output_variable = "hires"
        estimated_output_variable = "hires_estimate"
        ((i_h, i_w), (o_h, o_w), pattern) = test_spec
        folder = os.path.join(data_root_folder, pattern, f"{i_h}x{i_w}_{o_h}x{o_w}")

        if not os.path.exists(folder):
            print("No test data exists for this test.  Run script test/datagen/gen.py to generate the test data first")
            return

        train_path = os.path.join(folder, "train.nc")
        test_path = os.path.join(folder, "test.nc")
        mt = ConvAEModel(fc_size=16, encoded_dim_size=4, nr_epochs=500)
        mt.train(input_variable, output_variable, train_path, test_path)
        print(mt.summary())

        results_folder = os.path.join(results_root_folder, pattern, f"{i_h}x{i_w}_{o_h}x{o_w}")
        print("Writing test results to: " + results_folder)
        os.makedirs(results_folder, exist_ok=True)
        model_path = os.path.join(results_folder, "model")
        train_scores_path = os.path.join(results_folder, "train_scores.nc")
        test_scores_path = os.path.join(results_folder, "test_scores.nc")
        mt.save(model_path)

        mt2 = ConvAEModel()
        mt2.load(model_path)

        mt2.apply(train_path, input_variable, train_scores_path, estimated_output_variable)
        mt2.apply(test_path, input_variable, test_scores_path, estimated_output_variable)
        evaluation_html_path = os.path.join(results_folder, "model_evaluation.html")

        me = ModelEvaluator(train_scores_path, test_scores_path, input_variable, output_variable, evaluation_html_path,
                            estimated_output_variable, model_path)
        me.run()

if __name__ == '__main__':
    unittest.main()

