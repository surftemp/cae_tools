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
from torch import nn
import numpy as np
import xarray as xr
import json
import os
import time

from .model_sizer import create_model_spec


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


class Encoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size):
        super().__init__()

        encoder_layers = []
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                            stride=layer.get_stride()))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))

        self.encoder_cnn = nn.Sequential(*encoder_layers)

        self.flatten = nn.Flatten(start_dim=1)

        (chan, y, x) = layers[-1].get_output_dimensions()

        self.encoder_lin = nn.Sequential(
            nn.Linear(chan * y * x, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, layers, encoded_space_dim, fc_size):
        super().__init__()

        (chan, y, x) = layers[0].get_input_dimensions()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, chan * y * x),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, y, x))

        decoder_layers = []
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            decoder_layers.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                   stride=layer.get_stride(), output_padding=layer.get_output_padding()))
            if layer != layers[-1]:
                decoder_layers.append(nn.BatchNorm2d(output_channels))
                decoder_layers.append(nn.ReLU(True))

        self.decoder_conv = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class ConvAEModel:

    def __init__(self, input_variable, output_variable, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True):
        """
        Create a convolutional autoencoder general model

        :param input_variable: name of th input variable in training datasets
        :param output_variable: name of the output variable in training datasets
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
        """
        self.input_variable = input_variable
        self.output_variable = output_variable
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

        self.history = {}
        self.optim = None

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

        parameters = {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "batch_size": self.batch_size,
            "nr_epochs": self.nr_epochs,
            "test_interval": self.test_interval,
            "encoded_dim_size": self.encoded_dim_size,
            "fc_size": self.fc_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "normalise_input": self.normalise_input,
            "normalise_output": self.normalise_output
        }

        parameters_path = os.path.join(to_folder, "parameters.json")
        with open(parameters_path, "w") as f:
            f.write(json.dumps(parameters))

        history_path = os.path.join(to_folder, "history.json")
        with open(history_path, "w") as f:
            f.write(json.dumps(self.history))

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
            self.input_shape = tuple(parameters["input_shape"])
            self.output_shape = tuple(parameters["output_shape"])
            self.batch_size = parameters["batch_size"]
            self.nr_epochs = parameters["nr_epochs"]
            self.test_interval = parameters["test_interval"]
            self.encoded_dim_size = parameters["encoded_dim_size"]
            self.fc_size = parameters["fc_size"]
            self.lr = parameters["lr"]
            self.weight_decay = parameters["weight_decay"]
            self.normalise_input = parameters["normalise_input"]
            self.normalise_output = parameters["normalise_output"]

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        (input_chan, input_y, input_x) = self.input_shape
        (output_chan, output_y, output_x) = self.output_shape

        spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                 output_size=(output_y, output_x), output_channels=output_chan)
        print(spec)

        self.encoder = Encoder(spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        self.decoder = Decoder(spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder.eval()

    def train_epoch(self, batches):
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

    def test_epoch(self, batches, save_arr=None):
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

    def score(self, batches, save_arr):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                encoded_data = self.encoder(input_data)
                decoded_data = self.decoder(encoded_data)
                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

    def train(self, training_path, test_path):
        """
        Train the model (or continue training)

        :param training_path: path to a netcdf4 file containing input and output 4D arrays orgainsed by (N,CHAN,Y,X)
        :param test_path: path to a netcdf4 file to use for testing only.  Format as above
        """
        train_ds = DSDataset(xr.open_dataset(training_path), self.input_variable, self.output_variable,
                             normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds = DSDataset(xr.open_dataset(test_path), self.input_variable, self.output_variable,
                            normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                 output_size=(output_y, output_x), output_channels=output_chan)
        print(spec)

        self.encoder = Encoder(spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        self.decoder = Decoder(spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds.transform = train_transform
        test_ds.transform = test_transform

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size)

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

        self.history = {'train_loss': [], 'test_loss': []}

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

        for epoch in range(self.nr_epochs):
            train_loss = self.train_epoch(train_batches)
            if epoch % self.test_interval == 0:
                test_loss = self.test_epoch(test_batches)
                self.history["train_loss"].append(train_loss)
                self.history["test_loss"].append(test_loss)
                print("%5d %.6f %.6f" % (epoch, train_loss, test_loss))

        end = time.time()
        elapsed = end - start

        print("elapsed:" + str(elapsed))

    def predict(self, input_path, output_path, prediction_variable="prediction"):
        """
        Make predictions using this model

        :param input_path: path to a netcdf4 file containing input data
        :param output_path: path to a netcdf4 file to write containing the input data plus a prediction variable
        :param prediction_variable: the name of the prediction variable
        """
        score_ds = xr.open_dataset(input_path)
        score_arr = np.zeros(shape=score_ds[self.output_variable].shape)

        ds = DSDataset(score_ds, self.input_variable, self.input_variable, normalise_in=self.normalise_input)
        ds.set_normalisation_parameters(self.normalisation_parameters)
        val_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size)

        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        score_batches = []
        for low_res, _, _ in val_loader:
            low_res = low_res.to(device)
            score_batches.append(low_res)

        self.score(score_batches, save_arr=score_arr)
        score_ds[prediction_variable] = xr.DataArray(ds.denormalise_output(score_arr), dims=("n", "chan", "y", "x"))

        score_ds.to_netcdf(output_path)