import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import time


import numpy as np
import xarray as xr
import json
import os
import time

from .base_model import BaseModel
from .model_sizer import create_model_spec, ModelSpec
from .ds_dataset import DSDataset
from ..utils.model_database import ModelDatabase


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
    
class DualChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(DualChannelAttention, self).__init__()
        
        # global attention branch
        self.avg_pool_global = nn.AdaptiveAvgPool2d(1)
        self.fc1_global = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1_global = nn.ReLU()
        self.fc2_global = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        # spatial attention branch
        self.spatial_pool = nn.Conv2d(in_planes, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
        
        # activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global branch: learn attention that predicts the average
        avg_out = self.avg_pool_global(x)
        avg_out = self.fc2_global(self.relu1_global(self.fc1_global(avg_out)))
        
        # Spatial branch: learn attention based on spatial variations
        spatial_out = self.spatial_pool(x)
        spatial_out = self.spatial_sigmoid(spatial_out)
        
        # Combine the two branches (broadcasting spatial attention over channels)
        combined_attention = self.sigmoid(avg_out) * spatial_out
        
        return x * combined_attention    
    

class Encoder(nn.Module):
    def __init__(self, layers, encoded_space_dim, fc_size, dropout_rate=0.1):
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
    def __init__(self, layers, encoded_space_dim, fc_size, dropout_rate=0.1):
        super().__init__()

        (chan, y, x) = layers[0].get_input_dimensions()
        self.chan, self.y, self.x = layers[0].get_input_dimensions()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  #  dropout after ReLU            
            nn.Linear(fc_size, chan * y * x),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)  #  dropout after ReLU
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(chan, y, x))

        decoder_layers = []
        self.attention_layers = nn.ModuleList()
        for layer in layers:
            input_channels = layer.get_input_dimensions()[0]
            output_channels = layer.get_output_dimensions()[0]
            decoder_layers.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=layer.get_kernel_size(),
                                   stride=layer.get_stride(), padding=layer.get_output_padding()))
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

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[0,5, 10, 19, 28], device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers  # select layers to use for perceptual loss
        self.perceptual_encoder = nn.Sequential(*list(vgg)[:9]).to(device).eval()  # 
        for param in self.perceptual_encoder.parameters():
            param.requires_grad = False

        self.resize_transform = transforms.Resize((224, 224))
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])
        self.device = device

    def forward(self, predicted, ground_truth):
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

class UNET(BaseModel):
    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10, encoded_dim_size=32, fc_size=128,
                 lr=0.001, weight_decay=1e-5, dropout_rate=0.1, use_gpu=True, conv_kernel_size=3, conv_stride=2,
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
        self.dropout_rate = dropout_rate 
        self.use_gpu = use_gpu
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_input_layer_count = conv_input_layer_count
        self.conv_output_layer_count = conv_output_layer_count
        self.spec = None
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs': 0}
        self.optim = None
        self.optim_D = None
        self.db = ModelDatabase(database_path) if database_path else None
        self.lambda_l1 = lambda_l1
        self.lambda_pearson = lambda_pearson
        self.adversarial_loss = nn.BCELoss()
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.perceptual_loss_fn = VGGPerceptualLoss(device=self.device)  # Initialize perceptual loss component

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
            "lambda_pearson": self.lambda_pearson,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
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
    
    def __train_epoch(self, batches, device):
        self.encoder.train()
        self.decoder.train()
#         self.discriminator.train()
        lambda_l1 = self.lambda_l1
        lambda_pearson = self.lambda_pearson
        flag_pearson = False
        train_loss = []
        train_pearson_loss=[]
        train_bias_loss = []
        train_d_loss = []
        start_time = time.time()
        for i, (low_res, high_res, mask, labels) in enumerate(batches):
#             low_res = low_res.to(device, non_blocking=True)
#             high_res = high_res.to(device, non_blocking=True)
#             mask = mask.to(device, non_blocking=True).float()   
            
            self.optim.zero_grad()
            encoded_data, skip = self.encoder(low_res)
            decoded_data = self.decoder(encoded_data, skip)

            mse_loss = self.masked_mse_loss(decoded_data, high_res, mask)
            pearson_corr = self.pearson_corr_torch(decoded_data,high_res,mask)

            pearson_loss = 1 - torch.mean(pearson_corr)           
            
            combined_loss = mse_loss + lambda_pearson * pearson_loss #+ 0.1*bias_loss
            combined_loss.backward()
            self.optim.step()
            train_loss.append(mse_loss.item())
            train_pearson_loss.append(pearson_loss.item())
            #train_bias_loss.append(bias_loss.item())
            end_time = time.time()

        end_time = time.time()
        print(f"time used for training one epoch: {end_time - start_time:.2f}")
        mean_loss = np.mean(train_loss)
        mean_pearson_loss = np.mean(train_pearson_loss)

        mean_bias_loss = 0

        mean_d_loss = 0
        return float(mean_loss), float(mean_pearson_loss), 0, 0
    
    def __test_epoch(self, batches, device, save_arr=None):
        test_loss = []
        test_pearson_loss = []
        test_bias_loss = []
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  
            ctr = 0
            for (low_res, high_res, mask, labels) in batches:
#                 low_res = low_res.to(device, non_blocking=True)
#                 high_res = high_res.to(device, non_blocking=True)
#                 mask = mask.to(device, non_blocking=True).float()                
                encoded_data, skip = self.encoder(low_res)
                decoded_data = self.decoder(encoded_data, skip)

                loss = self.masked_mse_loss(decoded_data, high_res, mask)
                pearson_corr = self.pearson_corr_torch(decoded_data,high_res,mask)


                test_loss.append(loss.detach().cpu().numpy())

                pearson_loss = 1 - torch.mean(pearson_corr)
                test_pearson_loss.append(pearson_loss.detach().cpu().numpy())

                if save_arr is not None:
                    save_arr[ctr:ctr + self.batch_size, :, :, :] = decoded_data.cpu()
                ctr += self.batch_size

        mean_loss = np.mean(test_loss)
        mean_pearson_loss = np.mean(test_pearson_loss)
        mean_bias_loss = 0
        return float(mean_loss), float(mean_pearson_loss), float(mean_bias_loss)


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
        
    def train(self, input_variables, output_variable, training_ds, testing_ds, model_path="", training_paths="", testing_paths="",mask_variable_name=None):
        print("initiating train method")
        
        train_ds = DSDataset(training_ds, input_variables, output_variable,
                             normalise_in=self.normalise_input, normalise_out=self.normalise_output,
                             mask_variable_name=mask_variable_name)
        print("loaded train_ds to train method")
        self.set_input_spec(train_ds.get_input_spec())
        self.set_output_spec(train_ds.get_output_spec())
        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        
        test_ds = DSDataset(testing_ds, input_variables, output_variable,
                            normalise_in=self.normalise_input, normalise_out=self.normalise_output,
                            mask_variable_name=mask_variable_name)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)
        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)
        print("finished loading train_ds and test_ds from DSDataset")
        if not self.spec:
            self.spec = create_model_spec(input_size=(input_y, input_x), input_channels=input_chan,
                                 output_size=(output_y, output_x), output_channels=output_chan,
                                 kernel_size=self.conv_kernel_size, stride=self.conv_stride,
                                 input_layer_count=self.conv_input_layer_count, output_layer_count=self.conv_output_layer_count)

        if not self.encoder:
            self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate)
        if not self.decoder:
            self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate)
#         if not self.discriminator:
#             self.discriminator = Discriminator(output_chan)  # Ensure discriminator input channels match output image channels
        
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
        print("finished train_loarder and test_loader")
        if self.use_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")

        print(f'Running on device: {device}')

        start = time.time()

        self.loss_fn = torch.nn.MSELoss()
        self.encoder.to(device)
        self.decoder.to(device)
#         self.discriminator.to(device)

        self.optim = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        T_max=500
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=T_max, eta_min=self.lr)
        start_time = time.time()
#         train_batches = [(low_res.to(device), high_res.to(device), mask.to(device), labels)
#                            for low_res, high_res, mask, labels in train_loader]
#         test_batches = [(low_res.to(device), high_res.to(device), mask.to(device), labels)
#                           for low_res, high_res, mask, labels in test_loader]
        train_batches = []
        for low_res, high_res, mask, labels in train_loader: 
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            mask = mask.to(device)
            train_batches.append((low_res, high_res, mask, labels))
            
        test_batches = []
        for low_res, high_res, mask, labels in test_loader: 
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            mask = mask.to(device)
            test_batches.append((low_res, high_res, mask, labels))
            
#         train_batches = train_loader
#         test_batches = test_loader
        end_time = time.time()
        print(f"finished batching in {end_time - start_time:.2f} seconds")
        try:
            for epoch in range(self.nr_epochs):
                train_loss, train_pearson_loss, train_bias_loss, train_d_loss = self.__train_epoch(train_batches,device)
                if epoch<T_max:
                    scheduler.step()
                if epoch % self.test_interval == 0:
                    test_loss, test_pearson_loss, test_bias_loss = self.__test_epoch(test_batches,device)
#                     scheduler_D.step(test_loss)     
                    lr = self.get_lr(self.optim)
#                     lr_D = self.get_lr(self.optim_D)                    
                    self.history["train_loss"].append(train_loss)
                    self.history["test_loss"].append(test_loss)
                    print(f"epoch: {epoch}, train_mse: {train_loss:.6f}, train_pearson_loss: {train_pearson_loss:.4f}, test_mse: {test_loss:.6f}, test_pearson_loss: {test_pearson_loss:.4f}")
                    print(f"learn rate: {lr:.6f}")
        
#                 # early stopping condition
#                 if test_loss < 0.000447:
#                     print(f"Early stopping as Test MSE reached {test_loss:.6f}, below 0.000510.")
#                     break    
                    
        except KeyboardInterrupt:
            print("Training interrupted. Performing cleanup...")
        finally:
            end = time.time()
            elapsed = end - start

        self.history['nr_epochs'] += self.nr_epochs

        print("elapsed:" + str(elapsed))

        if self.db:
            self.db.add_training_result(self.get_model_id(), "UNET", output_variable, input_variables, self.summary(),
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

        self.encoder = Encoder(self.spec.get_input_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate)
        self.decoder = Decoder(self.spec.get_output_layers(), encoded_space_dim=self.encoded_dim_size, fc_size=self.fc_size,dropout_rate=self.dropout_rate)

        encoder_path = os.path.join(from_folder, "encoder.weights")
        self.encoder.load_state_dict(self.torch_load(encoder_path))
        self.encoder.eval()
        decoder_path = os.path.join(from_folder, "decoder.weights")
        self.decoder.load_state_dict(self.torch_load(decoder_path))
        self.decoder.eval()
        super().load(from_folder)
        
    def masked_mse_loss(self, pred, target, mask):
        diff = (pred - target) * mask
        mse_sum = torch.sum(diff ** 2)
        count = torch.sum(mask)
        return mse_sum / count          

    def pearson_corr_torch(self, decoded_data, high_res, mask):
        """
        Computes Pearson correlation per channel between decoded_data and high_res,
        using mask to consider only valid (unmasked) pixels.

        Assumes:
          - decoded_data and high_res are 4D tensors of shape (batch, channels, H, W)
          - mask is a 4D tensor of shape (batch, 1, H, W) or (batch, channels, H, W)
            If mask is (batch, 1, H, W), it will be broadcast over channels.
        """
        # flatten spatial dimensions: shape -> (batch, channels, N)
        decoded_data_flat = decoded_data.view(decoded_data.size(0), decoded_data.size(1), -1)
        high_res_flat = high_res.view(high_res.size(0), high_res.size(1), -1)
        mask_flat = mask.view(mask.size(0), mask.size(1), -1).float()

        # masked means:
        mean_decoded = torch.sum(decoded_data_flat * mask_flat, dim=2, keepdim=True) / (torch.sum(mask_flat, dim=2, keepdim=True) + 1e-8)
        mean_high_res = torch.sum(high_res_flat * mask_flat, dim=2, keepdim=True) / (torch.sum(mask_flat, dim=2, keepdim=True) + 1e-8)

        decoded_data_centered = decoded_data_flat - mean_decoded
        high_res_centered = high_res_flat - mean_high_res

        # masked standard deviations:
        std_decoded = torch.sqrt(torch.sum(mask_flat * (decoded_data_flat - mean_decoded) ** 2, dim=2, keepdim=True) /
                                   (torch.sum(mask_flat, dim=2, keepdim=True) + 1e-8) + 1e-8)
        std_high_res = torch.sqrt(torch.sum(mask_flat * (high_res_flat - mean_high_res) ** 2, dim=2, keepdim=True) /
                                   (torch.sum(mask_flat, dim=2, keepdim=True) + 1e-8) + 1e-8)

        # normalize:
        decoded_data_normalized = decoded_data_centered / std_decoded
        high_res_normalized = high_res_centered / std_high_res

        #get the masked pearson correlation: sum over unmasked pixels and divide by number of valid pixels.
        numerator = torch.sum(mask_flat * decoded_data_normalized * high_res_normalized, dim=2)
        denominator = torch.sum(mask_flat, dim=2)
        correlation = numerator / denominator

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
