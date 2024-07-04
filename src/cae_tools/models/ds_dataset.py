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

import torch
import numpy as np


class DSDataset(torch.utils.data.Dataset):

    def __init__(self, ds, input_variable_names, output_variable_name=None, normalise_in=True, normalise_out=True):
        self.ds = ds
        self.input_variable_names = input_variable_names
        self.output_variable_name = output_variable_name
        self.normalise_in = normalise_in
        self.normalise_out = normalise_out
        self.input_spec = []
        self.output_spec = None

        self.input_das = [self.ds[input_variable_name] for input_variable_name in input_variable_names]

        self.n = self.input_das[0].shape[0]
        self.input_chan = sum(input_da.shape[1] for input_da in self.input_das)
        self.input_y = self.input_das[0].shape[2]
        self.input_x = self.input_das[0].shape[3]

        # check for NaN
        da = self.ds[self.output_variable_name]
        count_nans = np.sum(np.where(np.isnan(da.values),1,0))
        if count_nans > 0:
            raise ValueError(f"output variable contains {count_nans} NaN values")

        # get the min/max for normalisation
        self.min_inputs = {}
        self.max_inputs = {}
        for idx in range(len(self.input_variable_names)):
            input_name = self.input_variable_names[idx]
            input_da = self.input_das[idx]
            self.min_inputs[input_name] = float(np.nanmin(input_da.values))
            self.max_inputs[input_name] = float(np.nanmax(input_da.values))
            count_nans = np.sum(np.where(np.isnan(input_da.values), 1, 0))
            if count_nans > 0:
                raise ValueError(f"input variable {input_name} contains {count_nans} NaN values")
            self.input_spec.append({"name":input_name,"shape":list(input_da.shape[1:])})

        if self.output_variable_name:
            self.output_da = self.ds[self.output_variable_name]
            self.output_chan = self.output_da.shape[1]
            self.output_y = self.output_da.shape[2]
            self.output_x = self.output_da.shape[3]
            self.min_output = float(np.nanmin(self.output_da.values))
            self.max_output = float(np.nanmax(self.output_da.values))
            self.output_spec = {"name":self.output_variable_name,"shape":list(self.output_da.shape[1:])}
        else:
            self.output_da = None
            self.output_chan = None
            self.output_y = None
            self.output_x = None
            self.min_output = None
            self.max_output = None

    def set_normalise_output(self, normalise_out):
        self.normalise_out = normalise_out

    def get_normalisation_parameters(self):
        return [self.min_inputs, self.max_inputs, self.min_output, self.max_output]

    def set_normalisation_parameters(self, parameters):
        (self.min_inputs, self.max_inputs, self.min_output, self.max_output) = tuple(parameters)

    def get_input_shape(self):
        return (self.input_chan, self.input_y, self.input_x)

    def get_input_spec(self):
        return self.input_spec

    def get_output_shape(self):
        return (self.output_chan, self.output_y, self.output_x)

    def get_output_spec(self):
        return self.output_spec

    def normalise_input(self, arr, input_name):
        if self.normalise_in:
            range_val = self.max_inputs[input_name] - self.min_inputs[input_name]
            if range_val == 0:
                return 0.0
            else:
                return (arr - self.min_inputs[input_name]) / range_val
        else:
            return arr

    def normalise_output(self, arr):
        if self.normalise_out:
            return (arr - self.min_output) / (self.max_output - self.min_output)
        else:
            return arr

    def denormalise_input(self, arr):
        if self.normalise_in:
            norm_arr = np.zeros(arr.shape, dtype=np.float32)
            channel_index = 0

            for idx in range(len(self.input_variable_names)):
                input_name = self.input_variable_names[idx]
                nchan = self.input_das[idx].shape[1]
                norm_arr[:, channel_index:channel_index + nchan, :, :] = \
                    self.min_inputs[input_name] + \
                        (arr[:, channel_index:channel_index + nchan, :,:] * \
                        (self.max_inputs[input_name] - self.min_inputs[input_name]))
            return norm_arr
        else:
            return arr

    def denormalise_output(self, arr, force=False):
        if force or self.normalise_out:
            return self.min_output + (arr * (self.max_output - self.min_output))
        else:
            return arr

    def __getitem__(self, index):
        label = f"image{index}"
        in_arr = np.zeros((self.input_chan,self.input_y,self.input_x), dtype=np.float32)
        channel_index = 0

        for idx in range(len(self.input_variable_names)):
            input_name = self.input_variable_names[idx]
            nchan = self.input_das[idx].shape[1]
            in_arr[channel_index:channel_index + nchan, :, :] = self.normalise_input(self.input_das[idx].data[index,:,:,:],input_name)
            channel_index += nchan

        if self.output_da is not None:
            out_arr = self.normalise_output(self.output_da[index, :, :, :].values)
        else:
            out_arr = None
        return (in_arr, out_arr, label)

    def __len__(self):
        return self.n