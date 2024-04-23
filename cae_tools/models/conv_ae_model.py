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
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import json
import os
import time

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from .encoder import Encoder
from .decoder import Decoder
from ..utils.model_database import ModelDatabase

class ConvAEModel(BaseModel):

    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None):
        """
        Create a convolutional autoencoder general model

        :param normalise_input: whether the input variable should be normalised
        :param normalise_output: whether the output variable should be normalised
        :param batch_size: batch size for training
        :param nr_epochs: number of iterations for training
        :param test_interval: calculate test statistics every this many iterations
        :param encoded_dim_size: size of the latent encoding, in neurons
        :param fc_size: size of the fully connected layers that connect the latent layer to encoder and decoder stages
        :param lr: learning rate
        :param weight_decay: weight decay?
        :param use_gpu: use GPU if present
        :param conv_kernel_size: size of the convolutional kernel to use
        :param conv_stride: stride to use in convolutional layers
        :param conv_input_layer_count: number of input convolutional layers to use
        :param conv_output_layer_count: number of output convolutional layers to use
        :param database_path: path to optional tracking database
        """
        super().__init__()
        self.normalise_input = normalise_input
        self.normalise_output = normalise_output
        self.normalisation_parameters = None
        self.input_shape = self.output_shape = None
        self.encoder = self.decoder = None
        self.batch_size = batch_size
        self.nr_epochs = nr_epochs
        self.test_interval = test_interval
        self.encoded_dim_size = encoded_dim_size
        self.fc_size = fc_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_input_layer_count = conv_input_layer_count
        self.conv_output_layer_count = conv_output_layer_count
        self.spec = None
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs':0 }
        self.optim = None
        self.db = ModelDatabase(database_path) if database_path else None

    def get_parameters(self):
        return {
            "type": "ConvAEModel",
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "batch_size": self.batch_size,
            "test_interval": self.test_interval,
            "encoded_dim_size": self.encoded_dim_size,
            "fc_size": self.fc_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "normalise_input": self.normalise_input,
            "normalise_output": self.normalise_output,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_stride": self.conv_stride,
            "conv_input_layer_count": self.conv_input_layer_count,
            "conv_output_layer_count": self.conv_output_layer_count,
            "model_id": self.get_model_id()
        }

    def save(self, to_folder):
        """
        Save the model to disk

        :param to_folder: folder to which model files are to be saved
        """
        os.makedirs(to_folder, exist_ok=True)
        encoder_path = os.path.join(to_folder, "encoder.weights")
        torch.save(self.encoder.state_dict(), encoder_path)
        decoder_path = os.path.join(to_folder, "decoder.weights")
        torch.save(self.decoder.state_dict(), decoder_path)
        normalisation_path = os.path.join(to_folder, "normalisation.weights")
        with open(normalisation_path, "w") as f:
            f.write(json.dumps(self.normalisation_parameters))

        parameters = self.get_parameters()

        parameters_path = os.path.join(to_folder, "parameters.json")
        with open(parameters_path, "w") as f:
            f.write(json.dumps(parameters))

        spec_path = os.path.join(to_folder, "spec.json")
        with open(spec_path, "w") as f:
            f.write(json.dumps(self.spec.save()))

        history_path = os.path.join(to_folder, "history.json")
        with open(history_path, "w") as f:
            f.write(json.dumps(self.history))

        summary_path = os.path.join(to_folder, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(self.summary())

    def load(self, from_folder):
        """
        Load a model from disk

        :param from_folder: folder from which model files should be loaded
        """
        normalisation_path = os.path.join(from_folder, "normalisation.weights")
        with open(normalisation_path, "r") as f:
            self.normalisation_parameters = json.loads(f.read())
        parameters_path = os.path.join(from_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())
            if "model_id" in parameters:
                self.set_model_id(parameters["model_id"])
            self.input_shape = tuple(parameters["input_shape"])
            self.output_shape = tuple(parameters["output_shape"])
            self.batch_size = parameters["batch_size"]
            self.test_interval = parameters["test_interval"]
            self.encoded_dim_size = parameters["encoded_dim_size"]
            self.fc_size = parameters["fc_size"]
            self.lr = parameters["lr"]
            self.weight_decay = parameters["weight_decay"]
            self.normalise_input = parameters["normalise_input"]
            self.normalise_output = parameters["normalise_output"]

            self.conv_kernel_size = parameters.get("conv_kernel_size",None)
            self.conv_stride = parameters.get("conv_stride",None)
            self.conv_input_layer_count = parameters.get("conv_input_layer_count",None)
            self.conv_output_layer_count = parameters.get("conv_output_layer_count",None)

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        spec_path = os.path.join(from_folder, "spec.json")
        with open(spec_path) as f:
            self.spec = ModelSpec()
            self.spec.load(json.loads(f.read()))

        self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder.eval()

    def __train_epoch(self, batches):
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        for (low_res, high_res, labels) in batches:
            # Encode data
            encoded_data = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data)
            loss = self.loss_fn(decoded_data, high_res)
            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Print batch loss
            # print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        mean_loss = np.mean(train_loss)
        return float(mean_loss)

    def __test_epoch(self, batches, save_arr=None):
        test_loss = []
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for (low_res, high_res, labels) in batches:
                # Encode data
                encoded_data = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data)
                loss = self.loss_fn(decoded_data, high_res)
                test_loss.append(loss.detach().cpu().numpy())
                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size
        mean_loss = np.mean(test_loss)
        return float(mean_loss)

    def __score(self, batches, save_arr):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                encoded_data = self.encoder(input_data)
                decoded_data = self.decoder(encoded_data)
                # Convert to CPU and then to NumPy for inspection
                encoded_data_np = encoded_data.cpu().numpy()
                decoded_data_np = decoded_data.cpu().numpy()

                # Debugging: Print or inspect these variables
                # print("encoded_data_np:", encoded_data_np)
                # print("decoded_data:", decoded_data_np)                   
                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

    def train(self, input_variables, output_variable, training_path, test_path, model_path=""):
        """
        Train the model (or continue training)

        :param input_variables: names of th input variables in training/test datasets
        :param output_variable: name of the output variable in training/test datasets
        :param training_path: path to a netcdf4 file containing input and output 4D arrays orgainsed by (N,CHAN,Y,X)
        :param test_path: path to a netcdf4 file to use for testing only.  Format as above
        :param model_path: path to save model to after training
        """
        train_ds = DSDataset(xr.open_dataset(training_path), input_variables, output_variable,
                             normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds = DSDataset(xr.open_dataset(test_path), input_variables, output_variable,
                            normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        if not self.spec:
            self.spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                 output_size=(output_y, output_x), output_channels=output_chan,
                                 kernel_size=self.conv_kernel_size, stride=self.conv_stride,
                                 input_layer_count=self.conv_input_layer_count, output_layer_count=self.conv_output_layer_count)

        if not self.encoder:
            self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        if not self.decoder:
            self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds.transform = train_transform
        test_ds.transform = test_transform

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        print(f'Running on device: {device}')

        start = time.time()

        self.loss_fn = torch.nn.MSELoss()

        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        self.optim = torch.optim.Adam(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)

        self.encoder.to(device)
        self.decoder.to(device)

        train_batches = []
        for low_res, high_res, labels in train_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            train_batches.append((low_res, high_res, labels))

        test_batches = []
        for low_res, high_res, labels in test_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            test_batches.append((low_res, high_res, labels))

        train_loss = test_loss = 0.0
        for epoch in range(self.nr_epochs):
            train_loss = self.__train_epoch(train_batches)
            if epoch % self.test_interval == 0:
                test_loss = self.__test_epoch(test_batches)
                self.history["train_loss"].append(train_loss)
                self.history["test_loss"].append(test_loss)
                print("%5d %.6f %.6f" % (epoch, train_loss, test_loss))

        end = time.time()
        elapsed = end - start

        self.history['nr_epochs'] = self.history['nr_epochs'] + self.nr_epochs

        print("elapsed:" + str(elapsed))

        if self.db:
            self.db.add_training_result(self.get_model_id(), "ConvAE", output_variable, input_variables, self.summary(),
                                        model_path, training_path, train_loss, test_path, test_loss, self.get_parameters(), self.spec.save())
        if model_path:
            self.save(model_path)

    def apply(self, input_path, input_variables, output_path, prediction_variable="model_output",
                channel_dimension="model_output_channel",y_dimension="model_output_y",x_dimension="model_output_x"):
        """
        Apply this model to input data to produce an output estimate

        :param input_path: path to a netcdf4 file containing input data
        :param input_variables: name of the input variables in the input data
        :param output_path: path to a netcdf4 file to write containing the input data plus a prediction variable
        :param prediction_variable: the name of the prediction variable
        :param channel_dimension: the name of the channel dimension in the prediction variable
        :param y_dimension: the name of the y dimension in the prediction variable
        :param x_dimension: the name of the x dimension in the prediction variable
        """
        score_ds = xr.open_dataset(input_path)
        # print("Input variables:", score_ds)
        # print("First item in input_variables:", input_variables[0])

        n = score_ds[input_variables[0]].shape[0]
        n_dimension = score_ds[input_variables[0]].dims[0]
        out_chan = self.output_shape[0]
        out_y = self.output_shape[1]
        out_x = self.output_shape[2]
        score_arr = np.zeros(shape=(n,out_chan,out_y,out_x))

        ds = DSDataset(score_ds, input_variables, input_variables[0], normalise_in=self.normalise_input)
        ds.set_normalisation_parameters(self.normalisation_parameters)
        val_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        self.encoder.to(device)
        self.decoder.to(device)

        score_batches = []
        for low_res, _, _ in val_loader:
            low_res = low_res.to(device)
            score_batches.append(low_res)

        self.__score(score_batches, save_arr=score_arr)
        score_ds[prediction_variable] = xr.DataArray(ds.denormalise_output(score_arr),
                dims=(n_dimension, channel_dimension, y_dimension, x_dimension))

        score_ds.to_netcdf(output_path)

    def summary(self):
        """
        Print a summary of the encoder/input and decoder/output layers
        """
        if self.spec:
            s = "Model Summary:\n"
            for input_spec in self.spec.input_layers:
                s += str(input_spec)
            s += "\tFully Connected Layer:\n"
            s += f"\t\tsize={self.fc_size}\n"
            s += "\tLatent Vector:\n"
            s += f"\t\tsize={self.encoded_dim_size}\n"
            s += "\tFully Connected Layer:\n"
            s += f"\t\tsize={self.fc_size}\n"
            for output_spec in self.spec.output_layers:
                s += str(output_spec)
            return s
        else:
            return "Model has not been trained - no layers assigned yet"
