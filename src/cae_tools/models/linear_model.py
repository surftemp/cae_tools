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
from .ds_dataset import DSDataset
from .linear import Linear
from ..utils.model_database import ModelDatabase

class LinearModel(BaseModel):

    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, database_path=None):
        """
        Create a simple linear model

        :param normalise_input: whether the input variable should be normalised
        :param normalise_output: whether the output variable should be normalised
        :param batch_size: batch size for training
        :param nr_epochs: number of iterations for training
        :param test_interval: calculate test statistics every this many iterations
        :param lr: learning rate
        :param weight_decay: weight decay?
        :param use_gpu: use GPU if present
        :param database_path: path to optional tracking database
        """

        self.normalise_input = normalise_input
        self.normalise_output = normalise_output
        self.normalisation_parameters = None
        self.input_shape = self.output_shape = None
        self.weights = None
        self.batch_size = batch_size
        self.nr_epochs = nr_epochs
        self.test_interval = test_interval
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs':0 }
        self.optim = None
        self.db = ModelDatabase(database_path) if database_path else None

    def get_parameters(self):
        return {
            "model_id": self.get_model_id(),
            "type": "LinearModel",
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "batch_size": self.batch_size,
            "test_interval": self.test_interval,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "normalise_input": self.normalise_input,
            "normalise_output": self.normalise_output
        }

    def save(self, to_folder):
        """
        Save the model to disk

        :param to_folder: folder to which model files are to be saved
        """
        os.makedirs(to_folder, exist_ok=True)
        weights_path = os.path.join(to_folder, "weights")
        torch.save(self.weights.state_dict(), weights_path)

        normalisation_path = os.path.join(to_folder, "normalisation.weights")
        with open(normalisation_path, "w") as f:
            f.write(json.dumps(self.normalisation_parameters))

        parameters = self.get_parameters()

        parameters_path = os.path.join(to_folder, "parameters.json")
        with open(parameters_path, "w") as f:
            f.write(json.dumps(parameters))

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
            self.lr = parameters["lr"]
            self.weight_decay = parameters["weight_decay"]
            self.normalise_input = parameters["normalise_input"]
            self.normalise_output = parameters["normalise_output"]

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        self.weights = Linear(self.input_shape,self.output_shape)

        weights_path = os.path.join(from_folder, "weights")
        self.weights.load_state_dict(torch.load(weights_path))
        self.weights.eval()


    def __train_epoch(self, batches):
        self.weights.train()

        train_loss = []
        for (low_res, high_res, labels) in batches:
            # Encode data
            estimates = self.weights(low_res)
            loss = self.loss_fn(estimates, high_res)
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
        self.weights.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for (low_res, high_res, labels) in batches:
                # Encode data
                estimates = self.weights(low_res)
                loss = self.loss_fn(estimates, high_res)
                test_loss.append(loss.detach().cpu().numpy())
                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = estimates.cpu()
                ctr += self.batch_size
        mean_loss = np.mean(test_loss)
        return float(mean_loss)

    def score(self, batches, save_arr):
        self.weights.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                estimates = self.weights(input_data)
                save_arr[ctr:ctr + self.batch_size, :, :, :] = estimates.cpu()
                ctr += self.batch_size

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
        super().__init__()
        train_ds = DSDataset(training_ds, input_variables, output_variable,
                             normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds = DSDataset(testing_ds, input_variables, output_variable,
                            normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        if not self.weights:
            self.weights = Linear(self.input_shape, self.output_shape)

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
            {'params': self.weights.parameters()}
        ]

        self.optim = torch.optim.Adam(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)

        self.weights.to(device)

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
            self.db.add_training_result(self.get_model_id(), "Linear", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss, test_paths, test_loss,
                                        self.get_parameters(), {})
        if model_path:
            self.save(model_path)

        # pass over the training and test sets and calculate model metrics

        metrics = {}
        metrics["test"] = self.evaluate(test_ds, device, self.batch_size)
        metrics["train"] = self.evaluate(train_ds, device, self.batch_size)
        self.dump_metrics("Test Metrics", metrics["test"])
        self.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.get_model_id(), training_paths, testing_paths, metrics)

    def apply(self, score_ds, input_variables, prediction_variable="model_output",
                channel_dimension="model_output_channel",y_dimension="model_output_y",x_dimension="model_output_x"):
        """
        Apply this model to input data to produce an output estimate, added to extend score_ds

        :param score_ds: an xarray dataset containing input data
        :param input_variables: name of the input variables in the input data
        :param prediction_variable: the name of the prediction variable
        :param channel_dimension: the name of the channel dimension in the prediction variable
        :param y_dimension: the name of the y dimension in the prediction variable
        :param x_dimension: the name of the x dimension in the prediction variable
        """
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

        self.weights.to(device)

        score_batches = []
        for low_res, _, _ in val_loader:
            low_res = low_res.to(device)
            score_batches.append(low_res)

        self.score(score_batches, save_arr=score_arr)
        score_ds[prediction_variable] = xr.DataArray(ds.denormalise_output(score_arr),
                dims=(n_dimension, channel_dimension, y_dimension, x_dimension))


    def summary(self):
        """
        Print a summary of the model
        """
        if self.input_shape:
            s = "Model Summary:\n"
            s += "\tInput shape:\n"
            s += f"\t\tsize={self.input_shape}\n"
            s += "\tOutput shape:\n"
            s += f"\t\tsize={self.output_shape}\n"
            return s
        else:
            return "Model has not been trained"
