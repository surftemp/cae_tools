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
import os
import json
import xarray as xr
import numpy as np

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.unet import UNET



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_paths", nargs="+", help="path to netcdf4 file(s) containing data to which model is applied")
    parser.add_argument("output_path", help="path to write the netcdf4 file containing input data plus model outputs")

    parser.add_argument("--model-folder", help="folder to save the trained model to", required=True)
    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=False)
    parser.add_argument("--prediction-variable", help="name of the prediction variable to create in output data",
                        default="model_output")
    parser.add_argument("--mask-variable", type=str, help="name of the mask variable", default=None)

    args = parser.parse_args()

    parameters_path = os.path.join(args.model_folder, "parameters.json")
    with open(parameters_path) as f:
        parameters = json.loads(f.read())

    if parameters["type"] == "ConvAEModel":
        mt = ConvAEModel()
    elif parameters["type"] == "UNET":
        mt = UNET()
    elif parameters["type"] == "LinearModel":
        mt = LinearModel()

    mt.load(args.model_folder)

    input_variable_names = args.input_variables
    if not input_variable_names:
        model_input_variable_names = mt.get_input_variable_names()

        if model_input_variable_names is None:
            raise Exception("Please specify the input variable names using --input-variables")
        else:
            input_variable_names = model_input_variable_names
    else:

        model_input_variable_names = mt.get_input_variable_names()
        if model_input_variable_names is not None:
            if input_variable_names != model_input_variable_names:
                raise Exception(f"input_variables [{','.join(input_variable_names)}] inconsistent with those used to train the model [{','.join(model_input_variable_names)}]")

    input_ds = xr.open_mfdataset(args.data_paths,concat_dim="box",combine="nested")
    case_dimension = input_ds[input_variable_names[0]].dims[0]
    score_ds = input_ds

    for var in args.input_variables:
        dims = score_ds[var].dims
        if dims == (case_dimension,):
            y_dim, x_dim = score_ds.dims['y'], score_ds.dims['x']


            original_values = score_ds[var].values  
            
            expanded_values = np.broadcast_to(original_values[:, np.newaxis, np.newaxis, np.newaxis], (original_values.shape[0], 1, y_dim, x_dim))
            expanded_var = xr.DataArray(expanded_values, coords={case_dimension: score_ds[case_dimension], 'channel': [0], 'y': np.arange(y_dim), 'x': np.arange(x_dim)}, dims=(case_dimension, 'channel', 'y', 'x'))

            score_ds[var] = expanded_var             
    
    print("Applying model for %d cases" % score_ds[case_dimension].shape[0])

    mt.apply(score_ds, input_variable_names, args.prediction_variable,mask_variable_name=args.mask_variable)
    score_ds.to_netcdf(args.output_path)

if __name__ == '__main__':
    main()