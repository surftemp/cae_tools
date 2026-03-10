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
from .linear import Linear, build_linear_model
from ..utils.model_database import ModelDatabase


class LinearModel(BaseModel):

    def __init__(self, normalise_input=True, normalise_output=True, batch_size=10,
                 nr_epochs=500, test_interval=10,
                 lr=0.001, weight_decay=1e-5, use_gpu=True, database_path=None,
                 architecture='pixel_linear', patience=None):
        """
        Create a simple linear model.

        Args:
            normalise_input: whether the input variable should be normalised
            normalise_output: whether the output variable should be normalised
            batch_size: batch size for training
            nr_epochs: number of iterations for training
            test_interval: calculate test statistics every this many iterations
            lr: learning rate
            weight_decay: weight decay
            use_gpu: use GPU if present
            database_path: path to optional tracking database
            architecture: model variant. One of:
                'full'         - original flatten+Linear (WARNING: huge for 100x100)
                'pixel_linear' - per-pixel linear via Conv2d(C,1,1). Default.
                'pixel_mlp'    - per-pixel MLP via stacked Conv2d 1x1 layers
        """
        super().__init__()
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
        self.architecture = architecture
        self.patience = patience
        self.history = {'train_loss': [], 'test_loss': [], 'nr_epochs': 0}
        self.optim = None
        self.loss_fn = torch.nn.MSELoss()
        self.db = ModelDatabase(database_path) if database_path else None

    def get_parameters(self):
        return {
            "model_id": self.get_model_id(),
            "type": "LinearModel",
            "architecture": self.architecture,
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

        super().save(to_folder)

    def load(self, from_folder):
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
            self.architecture = parameters.get("architecture", "full")

        history_path = os.path.join(from_folder, "history.json")
        with open(history_path) as f:
            self.history = json.loads(f.read())

        self.weights = build_linear_model(
            self.architecture, self.input_shape, self.output_shape)

        weights_path = os.path.join(from_folder, "weights")
        self.weights.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
        self.weights.eval()
        super().load(from_folder)

    def __train_epoch(self, batches, device=None):
            self.weights.train()
            train_loss = []
            train_mae = []
            for (low_res, high_res, labels) in batches:
                if device is not None:
                    low_res = low_res.to(device, non_blocking=True)
                    high_res = high_res.to(device, non_blocking=True)
                estimates = self.weights(low_res)
                loss = self.loss_fn(estimates, high_res)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss.append(loss.detach().cpu().numpy())
                with torch.no_grad():
                    train_mae.append((estimates - high_res).abs().mean().item())
            return float(np.mean(train_loss)), float(np.mean(train_mae))

    def __test_epoch(self, batches, device=None, save_arr=None):
            test_loss = []
            test_mae = []
            self.weights.eval()
            with torch.no_grad():
                ctr = 0
                for (low_res, high_res, labels) in batches:
                    if device is not None:
                        low_res = low_res.to(device, non_blocking=True)
                        high_res = high_res.to(device, non_blocking=True)
                    estimates = self.weights(low_res)
                    loss = self.loss_fn(estimates, high_res)
                    test_loss.append(loss.detach().cpu().numpy())
                    test_mae.append((estimates - high_res).abs().mean().item())
                    if save_arr is not None:
                        save_arr[ctr:ctr + self.batch_size, :, :, :] = estimates.cpu()
                    ctr += self.batch_size
            return float(np.mean(test_loss)), float(np.mean(test_mae))

    def score(self, batches, save_arr):
        self.weights.eval()
        with torch.no_grad():
            ctr = 0
            for input_data in batches:
                estimates = self.weights(input_data)
                save_arr[ctr:ctr + self.batch_size, :, :, :] = estimates.cpu()
                ctr += self.batch_size

    def train(self, input_variables, output_variable, training_ds, testing_ds,
              model_path="", training_paths="", testing_paths=""):
        """Train from xarray datasets (legacy .nc workflow)."""
        train_ds = DSDataset(training_ds, input_variables, output_variable,
                             normalise_in=self.normalise_input,
                             normalise_out=self.normalise_output)
        self.set_input_spec(train_ds.get_input_spec())
        self.set_output_spec(train_ds.get_output_spec())

        self.normalisation_parameters = train_ds.get_normalisation_parameters()

        test_ds = DSDataset(testing_ds, input_variables, output_variable,
                            normalise_in=self.normalise_input,
                            normalise_out=self.normalise_output)
        test_ds.set_normalisation_parameters(self.normalisation_parameters)

        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        if not self.weights:
            self.weights = build_linear_model(
                self.architecture, self.input_shape, self.output_shape)

        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_ds.transform = train_transform
        test_ds.transform = test_transform

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self._run_training(train_loader, test_loader, model_path)

    def train_from_datasets(self, train_ds, test_ds, model_path="",
                            training_paths="", testing_paths=""):
        """Train from PreprocessedDataset (.pt workflow).

        Args:
            train_ds: PyTorch Dataset with get_input_spec(), get_output_spec(),
                      get_normalisation_parameters(), get_input_shape(),
                      get_output_shape()
            test_ds: PyTorch Dataset (will use train_ds normalisation params)
            model_path: folder to save the trained model
            training_paths: string of training file paths (for logging)
            testing_paths: string of testing file paths (for logging)
        """
        self.set_input_spec(train_ds.get_input_spec())
        self.set_output_spec(train_ds.get_output_spec())
        self.normalisation_parameters = train_ds.get_normalisation_parameters()
        test_ds.set_normalisation_parameters(self.normalisation_parameters)

        (input_chan, input_y, input_x) = train_ds.get_input_shape()
        (output_chan, output_y, output_x) = train_ds.get_output_shape()

        self.input_shape = (input_chan, input_y, input_x)
        self.output_shape = (output_chan, output_y, output_x)

        n_params = sum(p.numel() for p in self.weights.parameters()) \
            if self.weights else '(not yet created)'
        print(f"Training cases: {len(train_ds)}, Test cases: {len(test_ds)}")
        print(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")
        print(f"Architecture: {self.architecture}")

        if not self.weights:
            self.weights = build_linear_model(
                self.architecture, self.input_shape, self.output_shape)

        n_params = sum(p.numel() for p in self.weights.parameters())
        print(f"Parameters: {n_params:,}")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=4,
                                  pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True)

        self._run_training(train_loader, test_loader, model_path)

    def _run_training(self, train_loader, test_loader, model_path):
            """Shared training loop with optional early stopping on test MAE."""
            if self.use_gpu:
                device = torch.device("cuda") if torch.cuda.is_available() \
                    else torch.device("cpu")
            else:
                device = torch.device("cpu")
            print(f"Running on device: {device}")
            if self.patience:
                print(f"Early stopping patience: {self.patience} epochs")
    
            self.weights.to(device)
    
            self.optim = torch.optim.Adam(
                self.weights.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)
    
            best_test_mae = float('inf')
            best_epoch = 0
            epochs_since_best = 0
    
            for epoch in range(self.nr_epochs):
                train_loss, train_mae = self.__train_epoch(train_loader, device)
                self.history['train_loss'].append(train_loss)
                self.history['nr_epochs'] = epoch + 1
    
                if epoch % self.test_interval == 0:
                    test_loss, test_mae = self.__test_epoch(test_loader, device)
                    self.history['test_loss'].append(test_loss)
    
                    out_range = self._get_output_range_k()
    
                    improved = ""
                    if test_mae < best_test_mae:
                        best_test_mae = test_mae
                        best_epoch = epoch
                        epochs_since_best = 0
                        self.history['best_test_mae'] = float(best_test_mae)
                        self.history['best_epoch'] = best_epoch
                        if model_path:
                            self.save(model_path)
                        improved = " ★ best"
                    else:
                        epochs_since_best += self.test_interval

                    print(f"epoch {epoch:4d}  "
                          f"train_mse={train_loss:.6f}  "
                          f"test_mse={test_loss:.6f}")
                    print(f"  metrics:    "
                          f"train_rmse={train_loss**0.5:.6f}  "
                          f"test_rmse={test_loss**0.5:.6f}  "
                          f"train_mae={train_mae:.6f}  "
                          f"test_mae={test_mae:.6f}  "
                          f"rmse/mae={test_loss**0.5/max(test_mae,1e-10):.3f}")
                    print(f"  physical:   "
                          f"train_rmse={train_loss**0.5*out_range:.2f}K  "
                          f"test_rmse={test_loss**0.5*out_range:.2f}K  "
                          f"train_mae={train_mae*out_range:.2f}K  "
                          f"test_mae={test_mae*out_range:.2f}K"
                          f"{improved}")

                    if self.patience and epochs_since_best >= self.patience:
                        print(f"\n  Early stopping: no improvement for "
                              f"{self.patience} epochs")
                        break

            print(f"\nTraining complete. Best test MAE: "
                  f"{best_test_mae*self._get_output_range_k():.2f}K "
                  f"at epoch {best_epoch}")

    def _get_output_range_k(self):
        """Get output range in Kelvin for converting normalised metrics."""
        if self.normalisation_parameters is None:
            return 1.0
        if isinstance(self.normalisation_parameters, list):
            min_out = list(self.normalisation_parameters[2].values())[0]
            max_out = list(self.normalisation_parameters[3].values())[0]
        else:
            min_out = self.normalisation_parameters.get('min_output', 0)
            max_out = self.normalisation_parameters.get('max_output', 1)
        return max_out - min_out

    def summary(self):
        if self.input_shape:
            s = f"LinearModel Summary (architecture={self.architecture}):\n"
            s += f"\tInput shape:  {self.input_shape}\n"
            s += f"\tOutput shape: {self.output_shape}\n"
            if self.weights:
                n = sum(p.numel() for p in self.weights.parameters())
                s += f"\tParameters:   {n:,}\n"
            return s
        else:
            return "Model has not been trained"