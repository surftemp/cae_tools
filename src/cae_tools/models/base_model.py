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

import uuid
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from .model_metric import ModelMetric

class BaseModel:

    def __init__(self):
        self.model_id = str(uuid.uuid4())

    def set_model_id(self, model_id):
        self.model_id = model_id

    def get_model_id(self):
        return self.model_id

    def evaluate(self, dataset, device, batch_size):

        # common code across the models to collect metrics

        dataset.set_normalise_output(False) # need to avoid normalising outputs when accessing the dataset

        dataset.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        mm = ModelMetric()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for input, output_not_norm, labels in loader:
            # for each batch, score, denormalise the scores and compare with the original outputs
            input = input.to(device)
            score_arr = np.zeros(output_not_norm.shape)
            self.__score([input], save_arr=score_arr)
            output_not_norm = output_not_norm.numpy(force=True)
            score_arr = dataset.denormalise_output(score_arr,force=True)
            # feed the instances in each batch into the model metric accumulator
            for i in range(batch_size):
                mm.accumulate(output_not_norm[i,::],score_arr[i,::])

        return mm.get_metrics()

    def dump_metrics(self, title, metrics):
        print("\n"+title)
        for key in metrics:
            print(f"\t{key:30s}:{metrics[key]}")

    def __score(self, batches, save_arr):
        pass # implement in sub-class