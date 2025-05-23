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

import xarray as xr
import os
import shutil
import numpy as np
import seaborn as sns
import pandas as pd
import math
import json
import tempfile
import torch
import random

from ..utils.html5.html5_builder import Html5Builder
from ..utils.table_fragment import TableFragment
from ..utils.image_fragment import ImageFragment, InlineImageFragment
from ..utils.model_database import ModelDatabase
from ..utils.utils import anti_aliasing_style

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.var_ae_model import VarAEModel
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.unet import UNET

from cae_tools.models.ds_dataset import DSDataset

osm_wms_url="https://eocis.org/mapproxy/service?service=WMS&request=GetMap&layers=osm&styles=&format=image%2Fpng&transparent=false&version=1.1.1&width={WIDTH}&height={HEIGHT}&srs=EPSG%3A27700&bbox={XMIN},{YMIN},{XMAX},{YMAX}"

class ModelEvaluator:

    def __init__(self, training_paths, testing_paths, output_html_folder="", model_output_variable="", model_path="",
                 database_path="", input_variables=[], sample_count=None, x_coordinate="", y_coordinate="", time_coordinate=""):
        self.training_paths = training_paths if training_paths else []
        self.testing_paths = testing_paths if testing_paths else []
        self.output_html_folder = output_html_folder

        self.model_path = model_path
        self.model_output_variable = model_output_variable
        self.database_path = database_path
        self.db = ModelDatabase(database_path) if database_path else None

        self.input_variables = input_variables if input_variables is not None else []
        self.sample_count = sample_count
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.time_coordinate = time_coordinate

        if self.output_html_folder:
            self.output_html_path = os.path.join(self.output_html_folder,"index.html")


        parameters_path = os.path.join(self.model_path, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())

        if parameters["type"] == "ConvAEModel":
            self.model = ConvAEModel()
        elif parameters["type"] == "UNET":
            self.model = UNET()
        elif parameters["type"] == "VarAEModel":
            self.model = VarAEModel()
        elif parameters["type"] == "LinearModel":
            self.model = LinearModel()

        self.model.load(self.model_path)
        print(f"Evaluating model id={self.model.get_model_id()}")
        self.model_input_variables = self.model.get_input_variable_names()
        self.output_variable = self.model.get_output_variable_name()
        for input_variable in self.input_variables:
            if input_variable not in self.model_input_variables:
                raise Exception(f"requested {input_variable} is not a model input")

    def compute_measure(self, dataset, idx, measure):
        predicted = dataset[self.model_output_variable][idx, 0, :, :].values
        actual = dataset[self.output_variable][idx, 0, :, :].values
        if measure == "mae":
            return np.mean(np.abs(predicted - actual))
        elif measure == "mse":
            return np.mean(np.power(predicted - actual, 2))
        else:
            raise ValueError("Unknown measure: " + measure)

    def run(self):
        case_dimension, train_ds, test_ds, metrics = self.evaluate_model_metrics()

        if self.output_html_path:
            self.build_html(case_dimension, train_ds, test_ds, metrics)

    def evaluate_model_metrics(self):

        train_ds = [xr.open_dataset(training_path) for training_path in self.training_paths]
        test_ds = [xr.open_dataset(testing_path) for testing_path in self.testing_paths]

        case_dimension = train_ds[0][self.output_variable].dims[0] if len(train_ds) else test_ds[0][self.output_variable].dims[0]

        train_ds_count = len(train_ds)
        if train_ds_count == 0:
            train_ds = None
        elif train_ds_count == 1:
            train_ds = train_ds[0]
        else:
            train_ds = xr.concat(train_ds, dim=case_dimension)

        test_ds_count = len(test_ds)
        if test_ds_count == 0:
            test_ds = None
        elif test_ds_count == 1:
            test_ds = test_ds[0]
        else:
            test_ds = xr.concat(test_ds, dim=case_dimension)

        training_cases_count = 0 if train_ds is None else train_ds[self.output_variable].shape[0]
        testing_cases_count = 0 if test_ds is None else test_ds[self.output_variable].shape[0]
        print("Evaluating training cases: %d, test cases: %d" % (training_cases_count,testing_cases_count))

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        metrics = {}
        if test_ds is not None:
            test_dsdata = DSDataset(test_ds, self.model.get_input_variable_names(), self.model.get_output_variable_name(),
                                normalise_in=self.model.normalise_input, normalise_out=False)
            test_dsdata.set_normalisation_parameters(self.model.normalisation_parameters)
            metrics["test"] = self.model.evaluate(test_dsdata, device)
            self.model.dump_metrics("Test Metrics", metrics["test"])
        if train_ds is not None:
            train_dsdata = DSDataset(train_ds, self.model.get_input_variable_names(),
                                    self.model.get_output_variable_name(),
                                    normalise_in=self.model.normalise_input, normalise_out=False)
            train_dsdata.set_normalisation_parameters(self.model.normalisation_parameters)
            metrics["train"] = self.model.evaluate(train_dsdata, device)
            self.model.dump_metrics("Train Metrics", metrics["train"])

        if self.db:
            self.db.add_evaluation_result(self.model.get_model_id(), ",".join(self.training_paths), ",".join(self.testing_paths), metrics)

        return case_dimension, train_ds, test_ds, metrics

    def build_html(self, case_dimension, train_ds, test_ds, model_metrics):
        # if the dataset does not already contain scores, apply the model to generate them into the dataset
        for (partition,ds) in [("train",train_ds), ("test",test_ds)]:
            if ds:
                if self.model_output_variable not in ds:
                    print(f"Applying model to generate {partition} scores")
                    self.model.apply(ds,input_variables=self.model.get_input_variable_names(),prediction_variable=self.model_output_variable)

        # get a list of variable names in the data
        ds = train_ds if train_ds is not None else test_ds
        variable_names = set()
        if ds:
            for name in ds.variables:
                variable_names.add(name)


        image_folder = os.path.join(self.output_html_folder, "images")
        os.makedirs(image_folder,exist_ok=True)


        builder = Html5Builder(language="en")

        builder.head().add_element("title").add_text("Model Evaluation")
        builder.head().add_element("style").add_text(anti_aliasing_style)

        builder.body().add_element("h2", {"id": "heading"}).add_text("Model Metrics")
        for (label,key) in [("Test Metrics","test"),("Train Metrics","train")]:
            if key in model_metrics:
                builder.body().add_element("h3").add_text(label)
                parameter_tbl = TableFragment()
                parameter_tbl.add_row(["Metric Name", "Metric Value"])
                for (k,v) in model_metrics[key].items():
                    parameter_tbl.add_row([k, f"{v:0.3f}"])
                builder.body().add_fragment(parameter_tbl)

        builder.body().add_element("h2", {"id": "heading"}).add_text("Model Evaluation Results")

        training_losses = None
        training_parameters = None
        if self.model_path:
            with open(os.path.join(self.model_path,"history.json")) as f:
                training_losses = json.loads(f.read())
            with open(os.path.join(self.model_path,"parameters.json")) as f:
                training_parameters = json.loads(f.read())

        # work out target/predicted min and max
        target_vmin = None
        target_vmax = None
        # and for each input variable
        input_vmins = {}
        input_vmaxes = {}
        for ds in [train_ds,test_ds]:
            if ds is not None:
                for v in [self.output_variable,self.model_output_variable]:
                    tmin = float(ds[v].min(skipna=True))
                    tmax = float(ds[v].max(skipna=True))
                    if target_vmin is None or tmin < target_vmin:
                        target_vmin = tmin
                    if target_vmax is None or tmax > target_vmax:
                        target_vmax = tmax

                for v in self.input_variables:
                    tmin = float(ds[v].min(skipna=True))
                    tmax = float(ds[v].max(skipna=True))
                    if v not in input_vmins or tmin < input_vmins[v]:
                        input_vmins[v] = tmin
                    if v not in input_vmaxes or tmax > input_vmaxes[v]:
                        input_vmaxes[v] = tmax

        converter_config = None
        if self.x_coordinate and self.y_coordinate and self.time_coordinate:
            converter_config = {
                "dimensions": {
                    "case": case_dimension
                },
                "coordinates": {
                    "x": self.x_coordinate,
                    "y": self.y_coordinate,
                    "time": self.time_coordinate
                },
                "image": {
                    "grid-width": 250,
                    "max-zoom": 10
                },
                "layers": {
                }
            }

            for v in self.input_variables + [self.output_variable, self.model_output_variable]:
                if v in self.input_variables:
                    vmin = input_vmins[v]
                    vmax = input_vmaxes[v]
                else:
                    vmin = target_vmin
                    vmax = target_vmax

                converter_config["layers"][v] = {
                    "label": v,
                    "type": "single",
                    "min_value": vmin,
                    "max_value": vmax,
                    "cmap": "coolwarm"
                }


        for (partition,ds) in [("test",test_ds),("train",train_ds)]:
            if ds is None:
                continue

            aux_inputs = []
            for vname in ds.variables:
                v = ds.variables[vname]
                if v.attrs.get("type","") == "auxilary-predictor":
                    aux_inputs.append(vname)

            builder.body().add_element("h3").add_text(partition)

            n = len(ds[case_dimension])

            all_measures = ["mae", "mse"]

            # first compute the measures for each case and add to the dataset

            for measure in all_measures:
                computed_measures = []
                for idx in range(0, n):
                    computed_measures.append(self.compute_measure(ds,idx,measure))

                ds[measure] = xr.DataArray(computed_measures,dims=(case_dimension,))

            if all_measures:
                # draw histograms to show measure distributions

                for measure in all_measures:
                    with tempfile.NamedTemporaryFile(suffix=".png") as p:
                        values = ds[measure].data,
                        plot = sns.histplot(values)
                        plot.set_title(measure)
                        fig = plot.get_figure()
                        fig.savefig(p.name)
                        fig.clear()

                        builder.body().add_fragment(InlineImageFragment(p.name))

            case_output_folder=os.path.join(self.output_html_folder,partition)

            if converter_config:
                try:
                    from netcdf2html.api.netcdf2html_converter import Netcdf2HtmlConverter
                    converter = Netcdf2HtmlConverter(converter_config,ds,case_output_folder,title=partition,sample_count=self.sample_count)
                    converter.run()
                    builder.body().add_element("p")\
                        .add_element("a", {"href": partition+"/index.html"})\
                        .add_text(f"Case summary for partition {partition}")
                except:
                    print("Unable to create case summary")

            if not training_parameters:
                training_parameters = json.loads(ds.attrs["training_parameters"])

        if training_parameters or training_losses:
            builder.body().add_element("h2").add_text("Training Summary")

        if training_parameters:
            builder.body().add_element("h2").add_text("Training Parameters")

            parameter_tbl = TableFragment()
            parameter_tbl.add_row(["Parameter Name", "Parameter Value"])
            if training_losses:
                nr_epochs = training_losses["nr_epochs"]
                parameter_tbl.add_row(["total epochs", str(nr_epochs)])

            for (k,v) in training_parameters.items():
                parameter_tbl.add_row([k,str(v)])

            builder.body().add_fragment(parameter_tbl)

        if training_losses:

            all_losses = [(v, "train") for v in training_losses["train_loss"]] + [(v, "test") for v in training_losses["test_loss"]]

            data_plot = pd.DataFrame(
                {"log_loss": [math.log10(t[0]) for t in all_losses], "type": [t[1] for t in all_losses],
                 "test_iteration": [idx for idx in range(0, len(training_losses["train_loss"]))]
                                   + [idx for idx in range(0, len(training_losses["test_loss"]))]})
            plot = sns.lineplot(data_plot, x="test_iteration", y="log_loss", hue="type")

            plot.set_title("history")
            fig = plot.get_figure()
            with tempfile.NamedTemporaryFile(suffix=".png") as p:
                fig.savefig(p.name)
                fig.clear()
                builder.body().add_fragment(InlineImageFragment(p.name, w=768))

        with open(self.output_html_path,"w") as f:
            f.write(builder.get_html())





