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
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import time

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from .encoder_res_skip import Encoder
from .decoder_res_skip import Decoder
from ..utils.model_database import ModelDatabase

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=8):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        ])

#         # Print all the parameters
#         for name, param in self.named_parameters():
#             print(f"{name}: {param.shape}")

    def forward(self, x):
        for i, layer in enumerate(self.layers):
#             print(f"Before layer {i} ({layer.__class__.__name__}): {x.size()}")
            x = layer(x)
#             print(f"After layer {i} ({layer.__class__.__name__}): {x.size()}")
        return x


class RESUNET_GAN(BaseModel):

    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None,
                 discriminator_feature_maps=64):
        """
        Create a convolutional autoencoder GAN model
        """
        super().__init__()
        self.normalise_input = normalise_input
        self.normalise_output = normalise_output
        self.normalisation_parameters = None
        self.input_shape = self.output_shape = None
        self.encoder = self.decoder = self.discriminator = None
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
        self.optim_G = self.optim_D = None
        self.db = ModelDatabase(database_path) if database_path else None
        self.discriminator_feature_maps = discriminator_feature_maps

    def get_parameters(self):
        return {
            "type": "RESUNET_GAN",
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
            "discriminator_feature_maps": self.discriminator_feature_maps,
            "model_id": self.get_model_id()
        }

    def save(self, to_folder):
        """
        Save the model to disk
        """
        os.makedirs(to_folder, exist_ok=True)
        encoder_path = os.path.join(to_folder, "encoder.weights")
        torch.save(self.encoder.state_dict(), encoder_path)
        decoder_path = os.path.join(to_folder, "decoder.weights")
        torch.save(self.decoder.state_dict(), decoder_path)
        discriminator_path = os.path.join(to_folder, "discriminator.weights")
        torch.save(self.discriminator.state_dict(), discriminator_path)
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
            self.discriminator_feature_maps = parameters.get("discriminator_feature_maps", 64)

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        spec_path = os.path.join(from_folder, "spec.json")
        with open(spec_path) as f:
            self.spec = ModelSpec()
            self.spec.load(json.loads(f.read()))
        self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, num_size_preserving_layers=2)
        self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size, num_size_preserving_layers=2)
        self.discriminator = Discriminator(input_channels=self.input_shape[0], feature_maps=self.discriminator_feature_maps)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(self.torch_load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(self.torch_load(decoder_path))
        self.decoder.eval()
        discriminator_path = os.path.join(from_folder, "discriminator.weights")
        self.discriminator.load_state_dict(self.torch_load(discriminator_path))
        self.discriminator.eval()
        super().load(from_folder)

    def __train_epoch(self, batches):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        train_loss_G = []
        train_loss_D = []
        train_pixel_loss = []

        for (low_res, high_res, labels) in batches:
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)

            # Encode and decode data
            encoded_data, skip = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data, skip)

            # Generate adversarial labels
            validity = self.discriminator(decoded_data)

            valid = torch.ones(validity.size(), requires_grad=False).to(self.device)
            fake = torch.zeros(validity.size(), requires_grad=False).to(self.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optim_D.zero_grad()

            real_loss = self.adversarial_loss(self.discriminator(high_res), valid)
            fake_loss = self.adversarial_loss(self.discriminator(decoded_data.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            self.optim_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            self.optim_G.zero_grad()

            validity = self.discriminator(decoded_data)

            pixel_loss = self.pixelwise_loss(decoded_data, high_res)
            g_loss = self.adversarial_loss(validity, valid) + pixel_loss

            g_loss.backward()
            self.optim_G.step()

            train_loss_G.append(g_loss.item())
            train_loss_D.append(d_loss.item())
            train_pixel_loss.append(pixel_loss.item())

        mean_loss_G = np.mean(train_loss_G)
        mean_loss_D = np.mean(train_loss_D)
        mean_pixel_loss = np.mean(train_pixel_loss)
        return float(mean_loss_G), float(mean_loss_D), float(mean_pixel_loss)

    def __test_epoch(self, batches, save_arr=None):
        test_loss_G = []
        test_loss_D = []
        test_pixel_loss = []

        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        with torch.no_grad():
            ctr = 0
            for (low_res, high_res, labels) in batches:
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                # Encode and decode data
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)

                # Generate adversarial labels
                validity = self.discriminator(decoded_data)

                valid = torch.ones(validity.size(), requires_grad=False).to(self.device)
                fake = torch.zeros(validity.size(), requires_grad=False).to(self.device)

                pixel_loss = self.pixelwise_loss(decoded_data, high_res)
                g_loss = self.adversarial_loss(validity, valid) + pixel_loss

                real_loss = self.adversarial_loss(self.discriminator(high_res), valid)
                fake_loss = self.adversarial_loss(self.discriminator(decoded_data), fake)
                d_loss = (real_loss + fake_loss) / 2

                test_loss_G.append(g_loss.item())
                test_loss_D.append(d_loss.item())
                test_pixel_loss.append(pixel_loss.item())

                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

        mean_loss_G = np.mean(test_loss_G)
        mean_loss_D = np.mean(test_loss_D)
        mean_pixel_loss = np.mean(test_pixel_loss)
        return float(mean_loss_G), float(mean_loss_D), float(mean_pixel_loss)


    def score(self, batches, save_arr):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            ctr = 0
            for input_data in batches:
                input_data = input_data.to(self.device)
                encoded_data, skip = self.encoder(input_data)
                decoded_data = self.decoder(encoded_data, skip)
                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

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

        if not self.spec:
            self.spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                          output_size=(output_y, output_x), output_channels=output_chan,
                                          kernel_size=self.conv_kernel_size, stride=self.conv_stride,
                                          input_layer_count=self.conv_input_layer_count, output_layer_count=self.conv_output_layer_count)

        if not self.encoder:
            self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        if not self.decoder:
            self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        if not self.discriminator:
            self.discriminator = Discriminator(input_channels=self.output_shape[0], feature_maps=self.discriminator_feature_maps)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds.transform = train_transform
        test_ds.transform = test_transform

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        if self.use_gpu:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        print(f'Running on device: {self.device}')

        start = time.time()

        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.MSELoss()

        self.optim_G = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)

        train_batches = [(low_res.to(self.device), high_res.to(self.device), labels) for low_res, high_res, labels in train_loader]
        test_batches = [(low_res.to(self.device), high_res.to(self.device), labels) for low_res, high_res, labels in test_loader]

        for epoch in range(self.nr_epochs):
            train_loss_G, train_loss_D, train_pixel_loss = self.__train_epoch(train_batches)
            if epoch % self.test_interval == 0:
                test_loss_G, test_loss_D, test_pixel_loss = self.__test_epoch(test_batches)
                self.history["train_loss"].append((train_loss_G, train_loss_D, train_pixel_loss))
                self.history["test_loss"].append((test_loss_G, test_loss_D, test_pixel_loss))
                print(f"Epoch {epoch}: Generator Loss: {train_loss_G:.6f}, Discriminator Loss: {train_loss_D:.6f}, Pixel Loss: {train_pixel_loss:.6f}")

        end = time.time()
        elapsed = end - start

        self.history['nr_epochs'] = self.history['nr_epochs'] + self.nr_epochs

        print("Elapsed time: " + str(elapsed))

        if self.db:
            self.db.add_training_result(self.get_model_id(), "ConvAE_GAN", output_variable, input_variables, self.summary(),
                                        model_path, training_paths, train_loss_G, testing_paths, test_loss_G, self.get_parameters(), self.spec.save())
        if model_path:
            self.save(model_path)

        metrics = {}
        metrics["test"] = self.evaluate(test_ds, self.device)
        metrics["train"] = self.evaluate(test_ds, self.device)

        self.dump_metrics("Test Metrics", metrics["test"])
        self.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.get_model_id(), training_paths, testing_paths, metrics)



    def summary(self):
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


