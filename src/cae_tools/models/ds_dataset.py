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

    def __init__(self, ds, input_variable_name, output_variable_name="", normalise_in=True, normalise_out=True):
        self.ds = ds
        self.input_variable_name = input_variable_name
        self.output_variable_name = output_variable_name
        self.normalise_in = normalise_in
        self.normalise_out = normalise_out

        self.input_da = self.ds[self.input_variable_name]

        self.n = self.input_da.shape[0]
        self.input_chan = self.input_da.shape[1]
        self.input_y = self.input_da.shape[2]
        self.input_x = self.input_da.shape[3]

        # get the min/max for normalisation
        self.min_input = float(np.min(self.input_da.values))
        self.max_input = float(np.max(self.input_da.values))

        self.output_da = self.ds[self.output_variable_name]
        self.output_chan = self.output_da.shape[1]
        self.output_y = self.output_da.shape[2]
        self.output_x = self.output_da.shape[3]

        self.min_output = float(np.min(self.output_da.values))
        self.max_output = float(np.max(self.output_da.values))

    def get_normalisation_parameters(self):
        return [self.min_input, self.max_input, self.min_output, self.max_output]

    def set_normalisation_parameters(self, parameters):
        (self.min_input, self.max_input, self.min_output, self.max_output) = tuple(parameters)

    def get_input_shape(self):
        return (self.input_chan, self.input_y, self.input_x)

    def get_output_shape(self):
        return (self.output_chan, self.output_y, self.output_x)

    def normalise_input(self, arr):
        if self.normalise_in:
            return (arr - self.min_input) / (self.max_input - self.min_input)
        else:
            return arr

    def normalise_output(self, arr):
        if self.normalise_out:
            return (arr - self.min_output) / (self.max_output - self.min_output)
        else:
            return arr

    def denormalise_input(self, arr):
        if self.normalise_in:
            return self.min_input + (arr * (self.max_input - self.min_input))
        else:
            return arr

    def denormalise_output(self, arr):
        if self.normalise_out:
            return self.min_output + (arr * (self.max_output - self.min_output))
        else:
            return arr

    def __getitem__(self, index):
        label = f"image{index}"
        in_arr = self.normalise_input(self.input_da[index, :, :, :].values)
        out_arr = self.normalise_output(self.output_da[index, :, :, :].values)
        return (in_arr, out_arr, label)

    def __len__(self):
        return self.n