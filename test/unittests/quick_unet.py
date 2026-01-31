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

from cae_tools.models.unet import UNET

import test_specs

data_root_folder = os.path.join(os.path.split(__file__)[0], "..", "data")
results_root_folder = os.path.join(os.path.split(__file__)[0], "..", "results", "unet_quick")


class QuickUNETTest(unittest.TestCase):

    def test_circle(self):
        self.__run("circle")

    def __run(self, test_spec_name):
        test_spec = test_specs.all_specs[test_spec_name]
        input_variables = test_spec["inputs"]
        output_variable = test_spec["output"]
        estimated_output_variable = test_spec["output"] + "_estimate"
        (i_h, i_w) = test_spec["input_size"]
        (o_h, o_w) = test_spec["output_size"]
        hyperparameters = test_spec.get("hyperparameters", {})

        folder = os.path.join(data_root_folder, test_spec_name, f"{i_h}x{i_w}_{o_h}x{o_w}")

        if not os.path.exists(folder):
            print("No test data exists for this test. Run script test/datagen/gen.py to generate the test data first")
            return

        train_path = os.path.join(folder, "train.nc")
        test_path = os.path.join(folder, "test.nc")

        train_ds = xr.open_dataset(train_path)
        test_ds = xr.open_dataset(test_path)

        # Use UNET with minimal epochs for quick test
        mt = UNET(nr_epochs=10, batch_size=32, encoded_dim_size=32, fc_size=64)
        mt.train(input_variables, output_variable, train_ds, test_ds, 
                 training_paths=train_path, testing_paths=test_path)
        print(mt.summary())

        results_folder = os.path.join(results_root_folder, test_spec_name, f"{i_h}x{i_w}_{o_h}x{o_w}")
        print("Writing test results to: " + results_folder)
        os.makedirs(results_folder, exist_ok=True)
        model_path = os.path.join(results_folder, "model")
        train_scores_path = os.path.join(results_folder, "train_scores.nc")
        test_scores_path = os.path.join(results_folder, "test_scores.nc")
        mt.save(model_path)

        # Test loading
        mt2 = UNET()
        mt2.load(model_path)

        # Test applying
        train_scores_ds = xr.open_dataset(train_path)
        mt2.apply(train_scores_ds, input_variables, estimated_output_variable)
        train_scores_ds.to_netcdf(train_scores_path)

        test_scores_ds = xr.open_dataset(test_path)
        mt2.apply(test_scores_ds, input_variables, estimated_output_variable)
        test_scores_ds.to_netcdf(test_scores_path)

        print(f"Test completed successfully for {test_spec_name}")


if __name__ == '__main__':
    unittest.main()
