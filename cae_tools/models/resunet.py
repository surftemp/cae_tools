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
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import json
import os
import time
import inspect

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from .encoder_res_skip import Encoder
from .decoder_res_skip import Decoder
from ..utils.model_database import ModelDatabase

class RESUNET(BaseModel):

    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None,
                 lambda_l1=0.01, lambda_pearson=1,lambda_additional=0.5, additional_loss_type=None, bins=256, min_val=0, max_val=1, sigma=0.01):
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
        :param lambda_l1: weight for total variation loss
        :param lambda_pearson: weight for Pearson correlation loss
        :param additional_loss_type: type of additional loss to use ('contrastive', 'histogram', 'perceptual')
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
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs': 0}
        self.optim = None
        self.db = ModelDatabase(database_path) if database_path else None
        self.lambda_l1 = lambda_l1
        self.lambda_pearson = lambda_pearson
        self.lambda_additional=lambda_additional
        self.additional_loss_type = additional_loss_type
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins
        self.sigma = sigma
        
        # Initialize perceptual loss model if needed
        if self.additional_loss_type == 'perceptual':
            vgg = models.vgg19(pretrained=True).features
            self.perceptual_model = nn.Sequential(*list(vgg.children())[:35]).eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        # Print out all the parameters to confirm them
        if inspect.stack()[1].filename.endswith("train_cae.py"):
            print(f"RESUNET initialized with parameters:\n"
                  f"normalise_input: {self.normalise_input}\n"
                  f"normalise_output: {self.normalise_output}\n"
                  f"batch_size: {self.batch_size}\n"
                  f"nr_epochs: {self.nr_epochs}\n"
                  f"test_interval: {self.test_interval}\n"
                  f"encoded_dim_size: {self.encoded_dim_size}\n"
                  f"fc_size: {self.fc_size}\n"
                  f"lr: {self.lr}\n"
                  f"weight_decay: {self.weight_decay}\n"
                  f"use_gpu: {self.use_gpu}\n"
                  f"lambda_l1: {self.lambda_l1}\n"
                  f"lambda_pearson: {self.lambda_pearson}\n"
                  f"additional_loss_type: {self.additional_loss_type}")

    def get_parameters(self):
        return {
            "type": "RESUNET",
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
        super().save(to_folder)

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

        self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, num_size_preserving_layers=1)
        self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, num_size_preserving_layers=1)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder.eval()
        super().load(from_folder)        
        
    def contrastive_loss(self, y_true, y_pred):
        contrast_true = torch.max(y_true) - torch.min(y_true)
        contrast_pred = torch.max(y_pred) - torch.min(y_pred)
        return torch.abs(contrast_true - contrast_pred)
    
    def variance_loss(self,y_true, y_pred):
        # Compute the mean of y_true and y_pred
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)

        # Compute the variance of y_true and y_pred
        var_true = torch.mean((y_true - mean_true) ** 2)
        var_pred = torch.mean((y_pred - mean_pred) ** 2)

        # Compute the absolute difference between the variances
        loss = torch.abs(var_true - var_pred)

        return loss    

    def histogram_loss(self, y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        delta = (self.max_val - self.min_val) / self.bins

        # Create bin centers
        bin_centers = torch.linspace(self.min_val + delta / 2, self.max_val - delta / 2, self.bins).to(y_true.device)
        
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

        # Compute the differentiable histograms
        hist_true = torch.exp(-0.5 * ((y_true - bin_centers) / self.sigma) ** 2).sum(dim=0)
        hist_pred = torch.exp(-0.5 * ((y_pred - bin_centers) / self.sigma) ** 2).sum(dim=0)

        hist_true = hist_true / hist_true.sum()
        hist_pred = hist_pred / hist_pred.sum()

        # Compute the histogram loss
        hist_loss = torch.sum(torch.abs(hist_true - hist_pred))
        return hist_loss

    def perceptual_loss(self, y_true, y_pred):
        # Ensure 3 channels by replicating grayscale channels
        if y_true.size(1) == 1:
            y_true = y_true.repeat(1, 3, 1, 1)
            y_pred = y_pred.repeat(1, 3, 1, 1)

        # Normalize using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).to(y_true.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(y_true.device).view(1, 3, 1, 1)
        y_true = (y_true - mean) / std
        y_pred = (y_pred - mean) / std
        self.perceptual_model.to(y_true.device)

        y_true_features = self.perceptual_model(y_true)
        y_pred_features = self.perceptual_model(y_pred)
        return nn.functional.mse_loss(y_pred_features, y_true_features)

    def get_additional_loss(self, y_true, y_pred):
        if self.additional_loss_type is None:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
        elif self.additional_loss_type == 'contrastive':
            return self.contrastive_loss(y_true, y_pred)
        elif self.additional_loss_type == 'histogram':
            return self.histogram_loss(y_true, y_pred)
        elif self.additional_loss_type == 'perceptual':
            return self.perceptual_loss(y_true, y_pred)
        elif self.additional_loss_type == 'variance':
            return self.variance_loss(y_true, y_pred)          
        else:
            raise ValueError(f"Unknown additional loss type: {self.additional_loss_type}")

    def pearson_corr_torch(self, decoded_data, high_res):
        # flatten
        decoded_data_flat = decoded_data.view(decoded_data.size(0), decoded_data.size(1), -1)
        high_res_flat = high_res.view(high_res.size(0), high_res.size(1), -1)

        # compute the mean
        mean_decoded = torch.mean(decoded_data_flat, dim=2, keepdim=True)
        mean_high_res = torch.mean(high_res_flat, dim=2, keepdim=True)

        # subtracting the mean
        decoded_data_centered = decoded_data_flat - mean_decoded
        high_res_centered = high_res_flat - mean_high_res

        # compute standard deviations
        std_decoded = torch.std(decoded_data_centered, dim=2, keepdim=True)
        std_high_res = torch.std(high_res_centered, dim=2, keepdim=True)

        # normalize by dividing by the standard deviation
        decoded_data_normalized = decoded_data_centered / std_decoded
        high_res_normalized = high_res_centered / std_high_res

        # Pearson correlation
        correlation = torch.mean(decoded_data_normalized * high_res_normalized, dim=2)

        return correlation

    def tv_loss(self, x):
        """Calculate Total Variation Loss"""
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def __train_epoch(self, batches):
        self.encoder.train()
        self.decoder.train()
        lambda_l1 = self.lambda_l1
        lambda_pearson = self.lambda_pearson
        lambda_additional = self.lambda_additional
        train_loss = []
        mse_losses = []
        pearson_losses = []
        tv_losses = []
        additional_losses = []

        for (low_res, high_res, labels) in batches:
            encoded_data, skip = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data, skip)

            # mse loss
            loss = self.loss_fn(decoded_data, high_res)
            mse_losses.append(loss.detach().cpu().numpy())

            # compute Pearson correlation and Pearson loss
            pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
            pearson_loss = 1 - torch.mean(pearson_corr)
            pearson_losses.append(pearson_loss.detach().cpu().numpy())

#             # compute tv loss (for regularization)
#             tv_loss = self.tv_loss(decoded_data)
#             tv_losses.append(tv_loss.detach().cpu().numpy())

            # compute additional loss
            additional_loss = self.get_additional_loss(high_res, decoded_data)
            additional_losses.append(additional_loss.detach().cpu().numpy())

            # combined loss
            combined_loss = loss + lambda_pearson * pearson_loss + lambda_additional*additional_loss # + lambda_l1 * tv_loss 

            # Backward pass
            self.optim.zero_grad()
            combined_loss.backward()
            self.optim.step()

            # Append the combined loss to the train loss list
            train_loss.append(combined_loss.detach().cpu().numpy())

        mean_loss = np.mean(train_loss)
        mean_mse_loss = np.mean(mse_losses)
        mean_pearson_loss = np.mean(pearson_losses)
        mean_tv_loss = np.mean(tv_losses)
        mean_additional_loss = np.mean(additional_losses)

        return float(mean_loss), float(mean_mse_loss), float(mean_pearson_loss), float(mean_tv_loss), float(mean_additional_loss)


    def __test_epoch(self, batches, save_arr=None):
        test_loss = []
        mse_losses = []
        pearson_losses = []
        tv_losses = []
        additional_losses = []

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for (low_res, high_res, labels) in batches:
                # Encode data
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)

                # mse loss
                loss = self.loss_fn(decoded_data, high_res)
                mse_losses.append(loss.detach().cpu().numpy())

                # compute Pearson correlation and Pearson loss
                pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
                pearson_loss = 1 - torch.mean(pearson_corr)
                pearson_losses.append(pearson_loss.detach().cpu().numpy())

#                 # compute tv loss (for regularization)
#                 tv_loss = self.tv_loss(decoded_data)
#                 tv_losses.append(tv_loss.detach().cpu().numpy())

                # compute additional loss
                additional_loss = self.get_additional_loss(high_res, decoded_data)
                additional_losses.append(additional_loss.detach().cpu().numpy())

                # combined loss
                combined_loss = loss + self.lambda_pearson * pearson_loss  + additional_loss # + self.lambda_l1 * tv_loss
                test_loss.append(combined_loss.detach().cpu().numpy())

                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

        mean_loss = np.mean(test_loss)
        mean_mse_loss = np.mean(mse_losses)
        mean_pearson_loss = np.mean(pearson_losses)
        mean_tv_loss = np.mean(tv_losses)
        mean_additional_loss = np.mean(additional_losses)

        return float(mean_loss), float(mean_mse_loss), float(mean_pearson_loss), float(mean_tv_loss), float(mean_additional_loss)


    def score(self, batches, save_arr):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                encoded_data, skip = self.encoder(input_data)
                decoded_data = self.decoder(encoded_data, skip)
                # Convert to CPU and then to NumPy for inspection
                encoded_data_np = encoded_data.cpu().numpy()
                decoded_data_np = decoded_data.cpu().numpy()

                # Debugging: Print or inspect these variables
                # print("encoded_data_np:", encoded_data_np)
                # print("decoded_data:", decoded_data_np)                   
                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
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
            train_loss, train_mse_loss, train_pearson_loss, train_tv_loss, train_additional_loss = self.__train_epoch(train_batches)
            if epoch % self.test_interval == 0:
                test_loss, test_mse_loss, test_pearson_loss, test_tv_loss, test_additional_loss = self.__test_epoch(test_batches)
                self.history["train_loss"].append(train_loss)
                self.history["test_loss"].append(test_loss)
                print(f"{epoch:5d} Train Loss: {train_loss:.6f} (MSE: {train_mse_loss:.6f}, Pearson: {train_pearson_loss:.6f}, TV: {train_tv_loss:.6f}, Additional: {train_additional_loss:.6f}) "
                      f"Test Loss: {test_loss:.6f} (MSE: {test_mse_loss:.6f}, Pearson: {test_pearson_loss:.6f}, TV: {test_tv_loss:.6f}, Additional: {test_additional_loss:.6f}) ")

        end = time.time()
        elapsed = end - start

        self.history['nr_epochs'] = self.history['nr_epochs'] + self.nr_epochs

        print("elapsed:" + str(elapsed))

        if self.db:
            self.db.add_training_result(self.get_model_id(), "ConvAE", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss, testing_paths, test_loss, self.get_parameters(), self.spec.save())
        if model_path:
            self.save(model_path)

        # pass over the training and test sets and calculate model metrics

        metrics = {}
        metrics["test"] = self.evaluate(test_ds, device)
        metrics["train"] = self.evaluate(train_ds, device)

        self.dump_metrics("Test Metrics", metrics["test"])
        self.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.get_model_id(), training_paths, testing_paths, metrics)


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
