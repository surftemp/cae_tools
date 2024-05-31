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
import matplotlib.pyplot as plt

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from .vae_encoder import VAE_Encoder
from .decoder import Decoder
from ..utils.model_database import ModelDatabase

class VarAEModel(BaseModel):

    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None,database_path=None):
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
            "type": "VarAEModel",
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

        self.encoder = VAE_Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder.eval()
        super().load(from_folder)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def total_loss(self, reconstructed, original, mu, logvar):
        beta = 1e-6
        recon_loss = self.reconstruction_loss_fn(reconstructed, original)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta*kl_loss

    def reconstruction_loss_fn(self, reconstructed, original):
        return torch.nn.functional.mse_loss(reconstructed, original)        

    # def __train_epoch(self, batches):
    #     self.encoder.train()
    #     self.decoder.train()
    #     train_loss = []
    #     for (low_res, high_res, labels) in batches:
    #         mu, log_var = self.encoder(low_res)
    #         z = self.reparameterize(mu, log_var)  # Reparameterization step
    #         decoded_data = self.decoder(z)

    #         # Calculate total loss
    #         loss = self.total_loss(decoded_data, high_res, mu, log_var)

    #         # Backward pass
    #         self.optim.zero_grad()
    #         loss.backward()
    #         self.optim.step()
    #         train_loss.append(loss.detach().cpu().numpy())

    #     mean_loss = np.mean(train_loss)
    #     return float(mean_loss)

    def print_layer_details(self, model):
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                layer_type = type(layer).__name__
                input_channels = layer.in_channels
                output_channels = layer.out_channels
                kernel_size = layer.kernel_size
                stride = layer.stride
                print(f"{layer_type} - Input Channels: {input_channels}, Output Channels: {output_channels}, Kernel Size: {kernel_size}, Stride: {stride}")

            elif isinstance(layer, torch.nn.Linear):
                layer_type = type(layer).__name__
                input_features = layer.in_features
                output_features = layer.out_features
                print(f"{layer_type} - Input Features: {input_features}, Output Features: {output_features}")   

    def __train_epoch(self, batches,epoch):        

        save_dir = f"./VAE_Train_Images"

        self.encoder.train()
        self.decoder.train()     
        train_loss = []
        # train_loss_reconstruction = []       
        image_counter = 0  # Counter for image file names

      

        for batch_idx, (low_res, high_res, labels) in enumerate(batches):
            # Forward pass
            mu, log_var = self.encoder(low_res)
            z = self.reparameterize(mu, log_var)
            decoded_data = self.decoder(z)

            # Calculate total loss
            loss = self.total_loss(decoded_data, high_res, mu, log_var)
            # loss_record_reconstruction = self.reconstruction_loss_fn(decoded_data, high_res)

            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss.append(loss.detach().cpu().numpy())
            # train_loss_reconstruction.append(loss_record_reconstruction.detach().cpu().numpy())

            # Save decoded images
            if epoch % self.test_interval == 0 and batch_idx == 0:
                image_counter = self.save_decoded_images(decoded_data, low_res, high_res,  image_counter, save_dir,epoch)

        mean_loss = np.mean(train_loss)
        # mean_loss_reconstruction = np.mean(train_loss_reconstruction)
        # print(f"Epoch {epoch} - Mean Loss: {mean_loss} - Mean Reconstruction Loss: {mean_loss_reconstruction}")
        return float(mean_loss)    
    
    def save_decoded_images(self, decoded_data, low_res,  high_res, image_counter, save_dir, epoch):
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to CPU and numpy arrays
        decoded_images = decoded_data.cpu().detach().numpy()
        high_res_images = high_res.cpu().detach().numpy()
        low_res_images = low_res.cpu().detach().numpy()

        # Loop through the images in the batch and save each
        for i in range(decoded_images.shape[0]):
            plt.figure(figsize=(6, 18))  # Adjust size as needed

            # Plot the decoded image
            plt.subplot(3, 1, 1)  # 3 row, 1 columns, 1st subplot
            plt.pcolormesh(decoded_images[i][0], cmap='jet')  # Adjust index for your data's structure
            plt.colorbar()
            plt.title(f"Epoch {epoch} Decoded Image {image_counter}")

            # Plot the corresponding high resolution image
            plt.subplot(3, 1, 2)  # 3 row, 1 columns, 2nd subplot
            plt.pcolormesh(high_res_images[i][0], cmap='jet')  # Adjust index for your data's structure
            plt.colorbar()
            plt.title(f"Epoch {epoch} Original Image {image_counter}")

            # Plot the corresponding high resolution image
            plt.subplot(3, 1, 3)  # 3 row, 2 columns, 3rd subplot
            plt.pcolormesh(low_res_images[i][0], cmap='jet')  # Adjust index for your data's structure
            plt.colorbar()
            plt.title(f"Epoch {epoch} Low Res Image {image_counter}")            

            plt.savefig(os.path.join(save_dir, f"Epoch_{epoch}_comparison_{image_counter}.png"))
            plt.close()
            image_counter += 1  # Increment the image counter

        return image_counter


    
    def __test_epoch(self, batches, save_arr=None):
        test_loss = []
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ctr = 0
            for (low_res, high_res, labels) in batches:
                mu, log_var = self.encoder(low_res)
                z = self.reparameterize(mu, log_var)
                decoded_data = self.decoder(z)
                loss = self.reconstruction_loss_fn(decoded_data, high_res)  # Just reconstruction loss
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
                mu, log_var = self.encoder(input_data)
                z = self.reparameterize(mu, log_var)  # Reparameterization step
                decoded_data = self.decoder(z)
                # # Convert to CPU and then to NumPy for inspection
                # mu_np = mu.cpu().numpy()
                # log_var_np = log_var.cpu().numpy()
                # z_np = z.cpu().numpy()
                # decoded_data_np = decoded_data.cpu().numpy()

                # # Debugging: Print or inspect these variables
                # print("mu:", mu_np)
                # print("log_var:", log_var_np)
                # print("z:", z_np) 
                # print("decoded_data:", decoded_data_np)       

                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

    def train(self, input_variables, output_variable, training_ds, testing_ds, model_path="", training_paths="",
              testing_paths=""):
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
            self.encoder = VAE_Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
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
        print("Encoder Layers:")
        self.print_layer_details(self.encoder)

        print("\nDecoder Layers:")
        self.print_layer_details(self.decoder)           

        start = time.time()

        # self.loss_fn = torch.nn.MSELoss()

        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        self.optim = torch.optim.Adam(params_to_optimize, lr=self.lr,betas=(0.9, 0.999), weight_decay=self.weight_decay)

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
            train_loss = self.__train_epoch(train_batches,epoch)
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
            self.db.add_training_result(self.get_model_id(), "VarAE", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss, testing_paths, test_loss,
                                        self.get_parameters(), self.spec.save())
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