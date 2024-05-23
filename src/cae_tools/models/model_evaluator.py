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
from PIL import Image
from matplotlib import cm
import numpy as np
import seaborn as sns
import datetime
import pandas as pd
import math
import json
import tempfile
import torch

from ..utils.html5.html5_builder import Html5Builder
from ..utils.table_fragment import TableFragment
from ..utils.image_fragment import ImageFragment, InlineImageFragment
from ..utils.utils import add_exo_dependencies
from ..utils.tab_fragment import TabbedFragment
from ..utils.model_database import ModelDatabase
from ..utils.utils import anti_aliasing_style

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.var_ae_model import VarAEModel
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.unet import UNET
from cae_tools.utils.layers import LayerFactory, LayerSingleBand

from cae_tools.models.ds_dataset import DSDataset

class ModelEvaluator:

    def __init__(self, training_paths, testing_paths, layer_config_path="", output_html_path="", model_output_variable="", model_path="",
                 database_path=""):
        self.training_paths = training_paths if training_paths else []
        self.testing_paths = testing_paths if testing_paths else []
        self.layer_config_path = layer_config_path
        self.output_html_path = output_html_path
        self.model_path = model_path
        self.model_output_variable = model_output_variable
        self.database_path = database_path
        self.db = ModelDatabase(database_path) if database_path else None

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
        self.input_variables = self.model.get_input_variable_names()
        self.output_variable = self.model.get_output_variable_name()

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

        image_folder = os.path.join(os.path.split(self.output_html_path)[0], "images")
        os.makedirs(image_folder,exist_ok=True)

        layer_definitions = []
        if self.layer_config_path:
            with open(self.layer_config_path) as f:
                layer_specs = json.loads(f.read())
                for (layer_name, layer) in layer_specs.items():
                    layer_definitions += LayerFactory.create(layer_name,layer,variable_names)

        builder = Html5Builder(language="en")
        add_exo_dependencies(builder.head())
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

        plot_variables = self.input_variables+[self.output_variable,self.model_output_variable]

        image_width=256

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

        if len(layer_definitions) == 0:
            for v in self.input_variables+[self.output_variable,self.model_output_variable]:
                if v in self.input_variables:
                    vmin = input_vmins[v]
                    vmax = input_vmaxes[v]
                else:
                    vmin = target_vmin
                    vmax = target_vmax
                layer_definitions.append(LayerSingleBand(v,v,v,vmin,vmax,"coolwarm"))

        for (partition,ds) in [("test",test_ds),("train",train_ds)]:
            if ds is None:
                continue

            aux_inputs = []
            for vname in ds.variables:
                v = ds.variables[vname]
                if v.attrs.get("type","") == "auxilary-predictor":
                    aux_inputs.append(vname)

            builder.body().add_element("h3").add_text(partition)

            n = ds[self.input_variables[0]].shape[0]

            all_measures = ["mae", "mse"]

            # first compute the measures for each case
            computed_measures = []
            for idx in range(0, n):
                computed_measures.append((idx,{measure:self.compute_measure(ds,idx,measure) for measure in all_measures}))

            sort_measure = "mae"
            sort_ascending = False

            computed_measures = sorted(computed_measures,key=lambda t:t[1][sort_measure],reverse=not sort_ascending)

            if computed_measures:
                # draw histograms to show measure distributions
                tf = TabbedFragment(partition+"_measures")
                for measure in all_measures:
                    with tempfile.NamedTemporaryFile(suffix=".png") as p:
                        values = [t[1][measure] for t in computed_measures]
                        plot = sns.histplot(values)
                        plot.set_title(measure)
                        fig = plot.get_figure()
                        fig.savefig(p.name)
                        fig.clear()
                        tf.add_tab(measure,InlineImageFragment(p.name))
                builder.body().add_fragment(tf)
                table_measures = all_measures
            else:
                # dummy computed measures
                computed_measures = [(idx, {}) for idx in range(0, n)]
                table_measures = []

            layer_labels = []
            if "time" in ds:
                layer_labels.append("time")
            for layer_definition in layer_definitions:
                layer_labels.append(layer_definition.layer_label)

            tbl = TableFragment()
            tbl.add_row(layer_labels+table_measures+aux_inputs)

            for (idx,measure_values) in computed_measures:
                ds_slice = ds.isel(**{case_dimension:idx})
                cells = []
                if "time" in ds:
                    time = ds_slice["time"].astype(str)[0]
                    cells.append(str(time))
                for layer_definition in layer_definitions:
                    img_filename = f"{layer_definition.layer_name}_{idx}.png"
                    img_path = os.path.join(image_folder,img_filename)
                    with tempfile.NamedTemporaryFile(suffix=".png") as p:

                        layer_definition.build(ds_slice,img_path)
                        cells.append(ImageFragment(f"images/{img_filename}",w=image_width))

                for measure in table_measures:
                    cells.append("%0.3f"%measure_values[measure])

                for aux_input in aux_inputs:
                    cells.append("%0.3f"%float(ds[aux_input].values[idx]))

                tbl.add_row(cells)

            builder.body().add_element("div", {"class": "exo-tree", "role": "tree"}).add_element("div", {
                "is": "exo-tree", "label": "Case-by-case details"}).add_fragment(tbl)

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





