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

import uuid
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import os
import json

from .model_metric import ModelMetric
from .ds_dataset import DSDataset

class BaseModel:

    def __init__(self):
        self.input_spec = None
        self.output_spec = None
        self.model_id = str(uuid.uuid4())

    def set_input_spec(self, input_spec):
        self.input_spec = input_spec

    def get_input_spec(self):
        return self.input_spec

    def set_output_spec(self, output_spec):
        self.output_spec = output_spec

    def get_output_spec(self):
        return self.output_spec

    def get_input_variable_names(self):
        if self.input_spec is None:
            return None
        return [item["name"] for item in self.input_spec]

    def get_output_variable_name(self):
        if self.output_spec is None:
            return None
        return self.output_spec["name"]

    def set_model_id(self, model_id):
        self.model_id = model_id

    def get_model_id(self):
        return self.model_id

    def torch_load(self, from_path):
        if torch.cuda.is_available():
            return torch.load(from_path)
        else:
            return torch.load(from_path, map_location=torch.device('cpu'))

    def evaluate(self, dataset, device):

        # common code across the models to collect metrics

        if hasattr(self,"encoder"):
            self.encoder.to(device)
        if hasattr(self,"decoder"):
            self.decoder.to(device)           

        dataset.set_normalise_output(False) # need to avoid normalising outputs when accessing the dataset

        dataset.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        mm = ModelMetric()
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        for input, output_not_norm, mask, labels in loader:
            # for each batch, score, denormalise the scores and compare with the original outputs
            input = input.to(device)
            score_arr = np.zeros(output_not_norm.shape)
            self.score([input], save_arr=score_arr)
            output_not_norm = output_not_norm.numpy()
            score_arr = dataset.denormalise_output(score_arr,force=True)
            mask_np = mask.cpu().numpy() if mask is not None else np.ones_like(output_not_norm)            
            # feed the instances in each batch into the model metric accumulator
            batch_size = output_not_norm.shape[0]
            for i in range(batch_size):
                # print(f"step: {i} size of output_not_norm: {output_not_norm.shape} and score_arr: {score_arr.shape}, batch_size: {batch_size}")
                mm.accumulate(output_not_norm[i,::],score_arr[i,::],mask_np[i,:,:])

        return mm.get_metrics()

    def apply(self, score_ds, input_variables, prediction_variable="model_output",
                channel_dimension="model_output_channel",y_dimension="model_output_y",x_dimension="model_output_x", mask_variable_name=None):
        """
        Apply this model to input data to produce an output estimate, added to extend score_ds

        :param score_ds: an xarray dataset containing input data
        :param input_variables: name of the input variables in the input data
        :param prediction_variable: the name of the prediction variable
        :param channel_dimension: the name of the channel dimension in the prediction variable
        :param y_dimension: the name of the y dimension in the prediction variable
        :param x_dimension: the name of the x dimension in the prediction variable
        """

        # print("Input variables:", score_ds)
        # print("First item in input_variables:", input_variables[0])

        n = score_ds[input_variables[0]].shape[0]
        n_dimension = score_ds[input_variables[0]].dims[0]
        out_chan = self.output_shape[0]
        out_y = self.output_shape[1]
        out_x = self.output_shape[2]
        score_arr = np.zeros(shape=(n,out_chan,out_y,out_x))

        ds = DSDataset(score_ds, input_variables, input_variables[0], normalise_in=self.normalise_input,mask_variable_name=mask_variable_name)
        ds.set_normalisation_parameters(self.normalisation_parameters)
        val_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")
            
        if hasattr(self,"encoder"):
            self.encoder.to(device)
        if hasattr(self,"decoder"):
            self.decoder.to(device)
        if hasattr(self,"vae"):
            self.vae.to(device)
        if hasattr(self,"unet_res"):
            self.unet_res.to(device)   
        if hasattr(self,"srcnn_res"):
            self.srcnn_res.to(device)              

        score_batches = []
        for low_res,_, _, _ in val_loader:
            low_res = low_res.to(device)
            score_batches.append(low_res)

        self.score(score_batches, save_arr=score_arr)
        score_ds[prediction_variable] = xr.DataArray(ds.denormalise_output(score_arr),
                dims=(n_dimension, channel_dimension, y_dimension, x_dimension))

    def dump_metrics(self, title, metrics):
        print("\n"+title)
        for key in metrics:
            print(f"\t{key:30s}:{metrics[key]}")

    def score(self, batches, save_arr):
        pass # implement in sub-class

    def save(self, to_folder):
        if self.input_spec is not None:
            input_spec_path = os.path.join(to_folder, "input_spec.json")
            with open(input_spec_path,"w") as f:
                f.write(json.dumps(self.input_spec))
        if self.output_spec is not None:
            output_spec_path = os.path.join(to_folder, "output_spec.json")
            with open(output_spec_path,"w") as f:
                f.write(json.dumps(self.output_spec))

    def load(self, from_folder):
        input_spec_path = os.path.join(from_folder,"input_spec.json")
        if os.path.exists(input_spec_path):
            with open(input_spec_path) as f:
                self.input_spec = json.loads(f.read())
        output_spec_path = os.path.join(from_folder, "output_spec.json")
        if os.path.exists(output_spec_path):
            with open(output_spec_path) as f:
                self.output_spec = json.loads(f.read())

    def train(self, input_variables, output_variable, training_ds, testing_ds, model_path="", training_paths="", testing_paths=""):
        """
        Train the model (or continue training)

        :param input_variables: names of th input variables in training/test datasets
        :param output_variable: name of the output variable in training/test datasets
        :param training_ds: an xarray dataset containing input and output 4D arrays orgainsed by (N,CHAN,Y,X)
        :param testing_ds: an xarray dataset to use for testing only.  Format as above
        :param model_path: path to save model to after training
        :param training_paths: a string providing a lst of all the training data paths
        :param testing_paths: a string providing a list of all the test data paths
        """
        pass # implement in sub-class

    def summary(self):
        """
        Print a summary of the encoder/input and decoder/output layers
        """
        pass # implement in sub-class

    def get_parameters(self):
        pass # implement in sub-class
