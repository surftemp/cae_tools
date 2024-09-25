import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from .mean_predictor import MeanPredictorSingleChannel


import numpy as np
import xarray as xr
import json
import os
import time

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from ..utils.model_database import ModelDatabase


class UNET(BaseModel):
    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None, lambda_l1=0.001, lambda_pearson=1):
        """
        Create a model for predicting the mean value of the ground truth.

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
        self.encoder = None
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
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs': 0}
        self.optim = None
        self.db = ModelDatabase(database_path) if database_path else None
        self.lambda_l1 = lambda_l1
        self.lambda_pearson = lambda_pearson
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return {
            "type": "MeanValuePredictor",
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

    def __train_epoch(self, batches):
        self.encoder.train()
        train_loss = []

        for i, (low_res, high_res, labels) in enumerate(batches):
            self.optim.zero_grad()
            predicted_mean = self.encoder(low_res)
            true_mean = torch.mean(high_res, dim=(2, 3), keepdim=True)  # Mean over spatial dimensions (x, y)

            mse_loss = self.loss_fn(predicted_mean, true_mean)
            mse_loss.backward()
            self.optim.step()
            train_loss.append(mse_loss.item())

        mean_loss = np.mean(train_loss)
        return float(mean_loss)

    def __test_epoch(self, batches):
        test_loss = []
        self.encoder.eval()
        with torch.no_grad():  # No need to track the gradients
            for (low_res, high_res, labels) in batches:
                predicted_mean = self.encoder(low_res)
                true_mean = torch.mean(high_res, dim=(2, 3), keepdim=True)
                loss = self.loss_fn(predicted_mean, true_mean)
                test_loss.append(loss.detach().cpu().numpy())

        mean_loss = np.mean(test_loss)
        return float(mean_loss)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']    
    
    def train(self, input_variables, output_variable, training_ds, testing_ds, model_path="", training_paths="", testing_paths=""):
        train_ds = DSDataset(training_ds, input_variables, output_variable,
                             normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        self.set_input_spec(train_ds.get_input_spec())
        self.set_output_spec(train_ds.get_output_spec())

        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds = DSDataset(testing_ds, input_variables, output_variable,
                            normalise_in=self.normalise_input, normalise_out=self.normalise_output)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        # Directly instantiate the new encoder
        self.encoder = MeanPredictorSingleChannel(fc_size=self.fc_size, dropout_rate=0.5)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        device = self.device
        print(f'Running on device: {device}')

        start = time.time()

        self.loss_fn = torch.nn.MSELoss()
        self.encoder.to(device)

        self.optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=50, T_mult=1, eta_min=0)

        train_batches = [(low_res.to(device), high_res.to(device), labels) for low_res, high_res, labels in train_loader]
        test_batches = [(low_res.to(device), high_res.to(device), labels) for low_res, high_res, labels in test_loader]

        try:
            for epoch in range(self.nr_epochs):
                train_loss = self.__train_epoch(train_batches)
                scheduler.step()
                if epoch % self.test_interval == 0:
                    test_loss = self.__test_epoch(test_batches)
                    lr = self.get_lr(self.optim)
                    self.history["train_loss"].append(train_loss)
                    self.history["test_loss"].append(test_loss)
                    print(f"epoch: {epoch}, train_mse: {train_loss:.6f}, test_mse: {test_loss:.6f}")
                    print(f"learn rate: {lr:.6f}")
                    
        except KeyboardInterrupt:
            print("Training interrupted. Performing cleanup...")
        finally:
            end = time.time()
            elapsed = end - start

        self.history['nr_epochs'] += self.nr_epochs

        print("elapsed:" + str(elapsed))

        if self.db:
            self.db.add_training_result(self.get_model_id(), "MeanValuePredictor", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss, testing_paths, test_loss, self.get_parameters(), self.spec.save())
        if model_path:
            self.save(model_path)

        metrics = {}
        metrics["test"] = self.evaluate(test_ds, device)
        metrics["train"] = self.evaluate(train_ds, device)

        self.dump_metrics("Test Metrics", metrics["test"])
        self.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.get_model_id(), training_paths, testing_paths, metrics)

    def evaluate(self, dataset, device):
        """Evaluate the model on a dataset"""
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        losses = []
        with torch.no_grad():
            for (low_res, high_res, labels) in data_loader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                predicted_mean = self.encoder(low_res)
                true_mean = torch.mean(high_res, dim=(2, 3), keepdim=True)
                loss = self.loss_fn(predicted_mean, true_mean)
                losses.append(loss.item())
        return np.mean(losses)

    def score(self, batches, save_arr):
        self.encoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                predicted_mean = self.encoder(input_data)
                save_arr[ctr:ctr + self.batch_size] = predicted_mean.cpu().numpy()
                ctr += self.batch_size

    def summary(self):
        """Print a summary of the encoder layers"""
        if self.spec:
            s = "Model Summary:\n"
            s += "\tFully Connected Layer:\n"
            s += f"\t\tsize={self.fc_size}\n"
            s += "\tLatent Vector:\n"
            s += f"\t\tsize={self.encoded_dim_size}\n"
            s += "\tOutput Layer:\n"
            s += "\t\tsize=1\n"  # Predicting the mean value
            return s
        else:
            return "Model has not been trained - no layers assigned yet"

    def save(self, to_folder):
        """Save the model to disk"""
        os.makedirs(to_folder, exist_ok=True)
        encoder_path = os.path.join(to_folder, "encoder.weights")
        torch.save(self.encoder.state_dict(), encoder_path)
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
        super().save(to_folder)

    def load(self, from_folder):
        """Load a model from disk"""
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

            self.conv_kernel_size = parameters.get("conv_kernel_size", None)
            self.conv_stride = parameters.get("conv_stride", None)
            self.conv_input_layer_count = parameters.get("conv_input_layer_count", None)
            self.conv_output_layer_count = parameters.get("conv_output_layer_count", None)

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        spec_path = os.path.join(from_folder, "spec.json")
        with open(spec_path) as f:
            self.spec = ModelSpec()
            self.spec.load(json.loads(f.read()))

        # Update to use MeanPredictorSingleChannel
        self.encoder = MeanPredictorSingleChannel(fc_size=self.fc_size, dropout_rate=0.5)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(self.torch_load(encoder_path))
        self.encoder.eval()
        super().load(from_folder)

   
