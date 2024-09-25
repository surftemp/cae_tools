import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models

import numpy as np
import xarray as xr
import json
import os
import time

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from ..utils.model_database import ModelDatabase

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, 50, 50)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 25, 25)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (batch_size, 256, 12, 12)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (batch_size, 512, 6, 6)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # (batch_size, 1, 3, 3)
            nn.Flatten(),  # (batch_size, 9)
            nn.Linear(9, 1),  # (batch_size, 1)
            nn.Sigmoid()  # Output a value between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Encoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, fc_size, dropout_rate=0.3):
        super().__init__()

        encoder_layers = []
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                            stride=layer.get_stride(), padding=layer.get_output_padding()))
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.Dropout(dropout_rate))  # Add dropout after ReLU

        self.encoder_cnn = nn.ModuleList(encoder_layers)
        self.flatten = nn.Flatten(start_dim=1)

        (chan, y, x) = layers[-1].get_output_dimensions()

        self.encoder_lin = nn.Sequential(
            nn.Linear(chan * y * x, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Add dropout after ReLU            
            nn.Linear(fc_size, encoded_space_dim),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)  # Add dropout after ReLU
        )

    def forward(self, x):
        x_skip = []
        for layer in self.encoder_cnn:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x_skip.append(x)

        x = self.flatten(x)
        x = self.encoder_lin(x)
        x_skip.pop()  # remove the last layer's output, not used for skip connections
        return x, x_skip

class Decoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, fc_size, dropout_rate=0.3):
        super().__init__()

        (chan, y, x) = layers[0].get_input_dimensions()
        self.chan, self.y, self.x = layers[0].get_input_dimensions()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Add dropout after ReLU            
            nn.Linear(fc_size, chan * y * x),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)  # Add dropout after ReLU
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, y, x))

        decoder_layers = []
        self.attention_layers = nn.ModuleList()
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            decoder_layers.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                   stride=layer.get_stride(), output_padding=layer.get_output_padding()))
            if layer != layers[-1]:
                self.attention_layers.append(ChannelAttention(output_channels))                
                decoder_layers.append(nn.BatchNorm2d(output_channels * 2))
                decoder_layers.append(nn.ReLU(True))
                decoder_layers.append(nn.Dropout(dropout_rate))  # Add dropout after ReLU

        self.decoder_conv = nn.ModuleList(decoder_layers)

    def forward(self, x, x_skip):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x_skip = x_skip[::-1]  # reverse to match decoder order

        skip_idx = 0        
        for layer in self.decoder_conv:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d) and skip_idx < len(x_skip):
                attention = self.attention_layers[skip_idx](x)
                x = x * attention  # Apply attention                
                x = torch.cat((x, x_skip[skip_idx]), 1)
                skip_idx += 1            
        x = torch.sigmoid(x)
        return x    

class UNET(BaseModel):
    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, conv_kernel_size=3, conv_stride=2,
                 conv_input_layer_count=None, conv_output_layer_count=None, database_path=None, lambda_l1=0.001, lambda_pearson=1):
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
        self.discriminator = None
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
        self.optim_G = None
        self.optim_D = None
        self.db = ModelDatabase(database_path) if database_path else None
        self.lambda_l1 = lambda_l1
        self.lambda_pearson = lambda_pearson
        self.adversarial_loss = nn.BCELoss()
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        self.init_perceptual_loss()  # Initialize perceptual loss components
    
    def init_perceptual_loss(self):
        # Initialize VGG19 model for perceptual loss
        vgg = models.vgg19(pretrained=True).features
        self.perceptual_encoder = nn.Sequential(*list(vgg)[:18])  # Use the first 18 layers of VGG19
        for param in self.perceptual_encoder.parameters():
            param.requires_grad = False
        self.perceptual_encoder.to(self.device)  # Move the perceptual encoder to the device
        
        self.resize_transform = transforms.Resize((224, 224))
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])

#     def compute_mean_std(self, data):
#         data = data.view(data.size(0), data.size(1), -1)
#         mean = data.mean(2).mean(0)
#         std = data.std(2).mean(0)
#         return mean, std

    def perceptual_loss(self, predicted, ground_truth):

#         self.data_mean, self.data_std = self.compute_mean_std(ground_truth)
#         self.normalize_transform = transforms.Normalize(mean=self.data_mean.tolist(), std=self.data_std.tolist())
        predicted_3channel = predicted.repeat(1, 3, 1, 1)
        ground_truth_3channel = ground_truth.repeat(1, 3, 1, 1)
        
        predicted_resized = self.resize_transform(predicted_3channel)
        ground_truth_resized = self.resize_transform(ground_truth_3channel)

        predicted_normalized = self.normalize_transform(predicted_resized)
        ground_truth_normalized = self.normalize_transform(ground_truth_resized)
        
        predicted_normalized = predicted_normalized.to(self.device)
        ground_truth_normalized = ground_truth_normalized.to(self.device)
        
        predicted_features = self.perceptual_encoder(predicted_normalized)
        ground_truth_features = self.perceptual_encoder(ground_truth_normalized)
        
        #  perceptual loss (MSE between VGG features)
        loss = nn.MSELoss()(predicted_features, ground_truth_features)
        return loss

    def get_parameters(self):
        return {
            "type": "UNET",
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

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype=torch.float32, requires_grad=True).to(real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size(), requires_grad=False).to(real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def __train_epoch(self, batches, n_critic=5):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        lambda_l1 = self.lambda_l1
        lambda_pearson = self.lambda_pearson
        flag_pearson = False
        train_loss = []
        train_pearson_loss=[]
        train_g_loss = []
        train_d_loss = []

        for i, (low_res, high_res, labels) in enumerate(batches):
            valid = torch.ones((high_res.size(0), 1 ), requires_grad=False).to(low_res.device)
            fake = torch.zeros((high_res.size(0), 1 ), requires_grad=False).to(low_res.device)

            # -----------------
            #  Train Generator
            # -----------------
            self.optim_G.zero_grad()
            encoded_data, skip = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data, skip)
            g_loss = self.adversarial_loss(self.discriminator(decoded_data), valid)
            mse_loss = self.loss_fn(decoded_data, high_res)
            pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
            pearson_loss = 1 - torch.mean(pearson_corr)
#             perceptual_loss = self.perceptual_loss(decoded_data, high_res) 
#             pearson_loss = perceptual_loss
            
            combined_loss = mse_loss + lambda_pearson * pearson_loss + g_loss
            combined_loss.backward()
            self.optim_G.step()
            train_loss.append(mse_loss.item())
            train_pearson_loss.append(pearson_loss.item())
            train_g_loss.append(g_loss.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if i % n_critic == 0:
                self.optim_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(high_res), valid)
                fake_loss = self.adversarial_loss(self.discriminator(decoded_data.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2 
                d_loss.backward()
                self.optim_D.step()
                train_d_loss.append(d_loss.item())

        mean_loss = np.mean(train_loss)
        mean_pearson_loss = np.mean(train_pearson_loss)
#         if pearson_loss <= 0.30:
#             flag_pearson = True
#         if flag_pearson:    
#             print(f"perceptual loss switched off at {pearson_loss}")
#             lambda_pearson = 0        
        mean_g_loss = np.mean(train_g_loss)
        mean_d_loss = np.mean(train_d_loss)
        return float(mean_loss), float(mean_pearson_loss), float(mean_g_loss), float(mean_d_loss)

    def __test_epoch(self, batches, save_arr=None):
        test_loss = []
        test_pearson_loss=[]
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for (low_res, high_res, labels) in batches:
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)
                pearson_corr = self.pearson_corr_torch(decoded_data, high_res)
                pearson_loss = 1 - torch.mean(pearson_corr)  
                test_pearson_loss.append(pearson_loss.detach().cpu().numpy())
                
                loss = self.loss_fn(decoded_data, high_res)
                test_loss.append(loss.detach().cpu().numpy())
                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size
                
        mean_loss = np.mean(test_loss)
        mean_pearson_loss = np.mean(test_pearson_loss)
        return float(mean_loss), float(mean_pearson_loss)

    def score(self, batches, save_arr):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                encoded_data, skip = self.encoder(input_data)
                decoded_data = self.decoder(encoded_data, skip)
                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

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
            self.discriminator = Discriminator(output_chan)  # Ensure discriminator input channels match output image channels
        
        fill_value = 0  
        fill = tuple([fill_value] * input_chan)
            
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),            # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),              # Randomly flip the image vertically
            transforms.RandomRotation(30, fill=fill),     # Randomly rotate the image within the range of -30 to +30 degrees, filling with the fill value
            transforms.RandomResizedCrop(100),            # Randomly crop the image and resize to 100x100
            transforms.ToTensor()
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
        self.encoder.to(device)
        self.decoder.to(device)
        self.discriminator.to(device)

        self.optim_G = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr/5, weight_decay=self.weight_decay)

#         scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(self.optim_G, mode='min', factor=0.1, patience=30, verbose=True)
#         scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(self.optim_D, mode='min', factor=0.1, patience=30, verbose=True)        

        train_batches = [(low_res.to(device), high_res.to(device), labels) for low_res, high_res, labels in train_loader]
        test_batches = [(low_res.to(device), high_res.to(device), labels) for low_res, high_res, labels in test_loader]

        try:
            for epoch in range(self.nr_epochs):
                train_loss, train_pearson_loss, train_g_loss, train_d_loss = self.__train_epoch(train_batches)
                if epoch % self.test_interval == 0:
                    test_loss, test_pearson_loss = self.__test_epoch(test_batches)
#                     scheduler_G.step(test_loss)
#                     scheduler_D.step(test_loss)     
                    lr_G = self.get_lr(self.optim_G)
                    lr_D = self.get_lr(self.optim_D)                    
                    self.history["train_loss"].append(train_loss)
                    self.history["test_loss"].append(test_loss)
                    print(f"epoch: {epoch}, train_mse: {train_loss:.6f}, train_pearson_loss: {train_pearson_loss:.4f}, test_mse: {test_loss:.6f}, test_pearson_loss: {test_pearson_loss:.4f}")
                    print(f"epoch: {epoch}, adversarial:g: {train_g_loss:.6f} d: {train_d_loss:.6f}")
                    print(f"learn rate: g: {lr_G:.6f}  d {lr_D:.6f}")
                    
        except KeyboardInterrupt:
            print("Training interrupted. Performing cleanup...")
        finally:
            end = time.time()
            elapsed = end - start

        self.history['nr_epochs'] += self.nr_epochs

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

    def score(self, batches, save_arr):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            ctr = 0
            for input_data in batches:
                encoded_data, skip = self.encoder(input_data)
                decoded_data = self.decoder(encoded_data, skip)
                save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

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

        self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)
        self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(self.torch_load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(self.torch_load(decoder_path))
        self.decoder.eval()
        super().load(from_folder)

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
