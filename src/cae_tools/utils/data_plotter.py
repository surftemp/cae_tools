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
from PIL import Image
from matplotlib import cm
import numpy as np
import datetime


from .html5.html5_builder import Html5Builder
from .table_fragment import TableFragment
from .image_fragment import InlineImageFragment
from .utils import add_exo_dependencies
import tempfile

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

class DataPlotter:

    def __init__(self, data_path, input_variables, output_variable, output_html_path):
        self.data_path = data_path

        self.input_variables = input_variables
        self.output_variable = output_variable
        self.output_html_path = output_html_path



    def run(self):
        builder = Html5Builder(language="en")
        add_exo_dependencies(builder.head())
        builder.head().add_element("title").add_text("Data Plot")
        builder.head().add_element("style").add_text(anti_aliasing_style)

        builder.body().add_element("h2", {"id": "heading"}).add_text(f"Plot for {self.data_path}")

        plot_variables = self.input_variables+[self.output_variable]

        image_width=256

        ds = xr.open_dataset(self.data_path)

        aux_inputs = []
        for vname in ds.variables:
            v = ds.variables[vname]
            if v.attrs.get("type","") == "auxilary-predictor":
                aux_inputs.append(vname)

        n = ds[self.input_variables[0]].shape[0]

        default_cmap = "coolwarm"

        tbl = TableFragment()
        tbl.add_row(["box"]+plot_variables+aux_inputs)

        vmins = {}
        vmaxs = {}
        for idx in range(0,n):
            cells = []
            cell_time = ds["box_time"][idx].data
            cells.append(str(cell_time)[:10])
            for target in plot_variables:
                with tempfile.NamedTemporaryFile(suffix=".png") as p:
                    splits=target.split(":")
                    name = splits[0]
                    if len(splits) > 1:
                        cmap = splits[1]
                    else:
                        cmap = default_cmap
                    if len(splits) > 2:
                        vmin = float(splits[2])
                    else:
                        if name not in vmins:
                            vmins[name] = float(ds[name].min())
                        vmin = vmins[name]


                    if len(splits) > 3:
                        vmax = float(splits[3])
                    else:
                        if name not in vmaxs:
                            vmaxs[name] = float(ds[name].max())
                        vmax = vmaxs[name]
                    print(name, vmin, vmax)
                    # save_image(ds[target][idx, 0, :, :], self.vmin, self.vmax, p.name, default_cmap)
                    save_image(ds[name][idx, 0, :, :], vmin, vmax, p.name, cmap)
                    cells.append(InlineImageFragment(p.name,w=image_width))

            for aux_input in aux_inputs:
                cells.append("%0.3f"%float(ds[aux_input].values[idx]))

            tbl.add_row(cells)

        builder.body().add_fragment(tbl)

        with open(self.output_html_path,"w") as f:
            f.write(builder.get_html())

