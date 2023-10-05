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

import pandas as pd
import math
import json

from htmlfive.html5_builder import Html5Builder
from .table_fragment import TableFragment
from .image_fragment import InlineImageFragment
from .utils import add_exo_dependencies
from .tab_fragment import TabbedFragment

from .utils import anti_aliasing_style

def save_image(arr,vmin,vmax,path,cmap_name):
    if cmap_name == "viridis":
        cmap_fn = cm.viridis
    elif cmap_name == "coolwarm":
        cmap_fn = cm.coolwarm
    else:
        raise ValueError("Unknown colour map: "+cmap_name)
    im = Image.fromarray(np.uint8((255*cmap_fn((arr-vmin)/(vmax-vmin)))))
    im.save(path)



class ModelEvaluator:

    def __init__(self, train_path, test_path, input_variable, output_variable, output_html_path, model_output_variable="", model_path=""):
        self.train_path = train_path
        self.test_path = test_path
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.output_html_path = output_html_path
        self.model_path = model_path
        self.model_output_variable = model_output_variable

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
        builder = Html5Builder(language="en")
        add_exo_dependencies(builder.head())
        builder.head().add_element("title").add_text("Model Evaluation")
        builder.head().add_element("style").add_text(anti_aliasing_style)

        builder.body().add_element("h2", {"id": "heading"}).add_text("Model Evaluation Results")

        training_losses = None
        training_parameters = None
        if self.model_path:
            with open(os.path.join(self.model_path,"history.json")) as f:
                training_losses = json.loads(f.read())
            with open(os.path.join(self.model_path,"parameters.json")) as f:
                training_parameters = json.loads(f.read())

        image_path = "/tmp/image.png"

        plot_variables = [self.input_variable,self.output_variable]
        if self.model_output_variable:
            plot_variables.append(self.model_output_variable)

        vmin = None
        vmax = None

        image_width=256

        for path in [self.test_path,self.train_path]:
            ds = xr.open_dataset(path)
            for v in plot_variables:
                tmin = float(ds[v].min(skipna=True))
                tmax = float(ds[v].max(skipna=True))
                if vmin is None or tmin < vmin:
                    vmin = tmin
                if vmax is None or tmax > vmax:
                    vmax = tmax

        for (partition,path) in [("test",self.test_path),("train",self.train_path)]:

            ds = xr.open_dataset(path)
            builder.body().add_element("h3").add_text(partition)

            n = ds[self.input_variable].shape[0]

            default_cmap = "coolwarm"

            all_measures = ["mae", "mse"]

            # first compute the measures
            computed_measures = []
            if self.model_output_variable:
                for idx in range(0, n):
                    computed_measures.append((idx,{measure:self.compute_measure(ds,idx,measure) for measure in all_measures}))

                sort_measure = "mae"
                sort_ascending = False

                computed_measures = sorted(computed_measures,key=lambda t:t[1][sort_measure],reverse=not sort_ascending)

            if computed_measures:
                tf = TabbedFragment(partition+"_measures")
                for measure in all_measures:
                    values = [t[1][measure] for t in computed_measures]
                    plot = sns.histplot(values)
                    plot.set_title(measure)
                    fig = plot.get_figure()
                    fig.savefig(image_path)
                    fig.clear()
                    tf.add_tab(measure,InlineImageFragment(image_path))
                builder.body().add_fragment(tf)

            table_measures = []
            if self.model_output_variable:
                table_measures = ["mae"]

            tbl = TableFragment()
            tbl.add_row(plot_variables+table_measures)
            if computed_measures:
                for (idx,measure_values) in computed_measures:
                    cells = []
                    for target in plot_variables:
                        save_image(ds[target][idx, 0, :, :], vmin, vmax, image_path, default_cmap)
                        cells.append(InlineImageFragment(image_path,w=image_width))

                    for measure in table_measures:
                        cells.append("%0.3f"%measure_values[measure])

                    tbl.add_row(cells)
            else:
                for idx in range(n):
                    cells = []
                    for target in plot_variables:
                        save_image(ds[target][idx, 0, :, :], vmin, vmax, image_path, default_cmap)
                        cells.append(InlineImageFragment(image_path, w=image_width))
                    tbl.add_row(cells)


            builder.body().add_element("div", {"class": "exo-tree", "role": "tree"}).add_element("div", {
                "is": "exo-tree", "label": "Details"}).add_fragment(tbl)

            if not training_parameters:
                training_parameters = json.loads(ds.attrs["training_parameters"])


        if training_parameters or training_losses:
            builder.body().add_element("h2").add_text("Training Summary")

        if training_parameters:
            builder.body().add_element("h2").add_text("Training Parameters")

            parameter_tbl = TableFragment()
            parameter_tbl.add_row(["Parameter Name", "Parameter Value"])
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
            fig.savefig(image_path)
            fig.clear()
            builder.body().add_fragment(InlineImageFragment(image_path, w=768))

        with open(self.output_html_path,"w") as f:
            f.write(builder.get_html())

