# MIT License
#
# Copyright (c) 2023 Niall McCarroll
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import requests
import fnmatch

from PIL import Image
from matplotlib import cm
import numpy as np

def save_image(arr,vmin,vmax,path,cmap_name="coolwarm"):
    if cmap_name == "viridis":
        cmap_fn = cm.viridis
    elif cmap_name == "coolwarm":
        cmap_fn = cm.coolwarm
    else:
        raise ValueError("Unknown colour map: "+cmap_name)
    im = Image.fromarray(np.uint8((255*cmap_fn((arr-vmin)/(vmax-vmin)))))
    im.save(path)

def save_image_falsecolour(data_red, data_green, data_blue, path):
    alist = []
    for arr in [data_red, data_green, data_blue]:
        minv = np.nanmin(arr)
        maxv = np.nanmax(arr)
        v = (arr - minv) / (maxv - minv)
        v = np.sqrt(v)
        alist.append((255*v).astype(np.uint8))
    arr = np.stack(alist,axis=-1)
    im = Image.fromarray(arr,mode="RGB")
    im.save(path)

def save_image_mask(arr, path, r, g, b):
    alist = []
    a = np.zeros(arr.shape)
    alist.append((a + r).astype(np.uint8))
    alist.append((a + g).astype(np.uint8))
    alist.append((a + b).astype(np.uint8))
    alist.append(np.where(arr>0,255,0).astype(np.uint8))
    rgba_arr = np.stack(alist, axis=-1)
    im = Image.fromarray(rgba_arr, mode="RGBA")
    im.save(path)


class LayerRGB:

    def __init__(self, layer_name, layer_label, red_variable, green_variable, blue_variable):
        self.layer_name = layer_name
        self.layer_label = layer_label
        self.red_variable = red_variable
        self.green_variable = green_variable
        self.blue_variable = blue_variable

    def has_legend(self):
        return False

    def check(self, ds):
        for variable in [self.red_variable, self.green_variable, self.blue_variable]:
            if variable not in ds:
                return f"No variable {variable}"

    def build(self,ds,path,flip):
        red = ds[self.red_variable].squeeze().data[:, :]
        green = ds[self.green_variable].squeeze().data[:, :]
        blue = ds[self.blue_variable].squeeze().data[:, :]

        if flip:
            red = np.flipid(red)
            green = np.flipud(green)
            blue = np.flipud(blue)
        save_image_falsecolour(red, green, blue, path)


class LayerSingleBand:

    def __init__(self, layer_name, layer_label, band_name, vmin, vmax, cmap_name):
        self.layer_name = layer_name
        self.layer_label = layer_label
        self.band_name = band_name
        self.vmin = vmin
        self.vmax = vmax
        self.cmap_name = cmap_name

    def check(self, ds):
        if self.band_name not in ds:
            return f"No variable {self.band_name}"

    def build(self,ds,path,flip=False):
        da = ds[self.band_name]
        arr = da.squeeze().data[:, :]
        if flip:
            arr = np.flipud(arr)
        save_image(arr, self.vmin, self.vmax, path, self.cmap_name)

    def has_legend(self):
        return True

    def build_legend(self, path):
        legend_width, legend_height = 200, 20
        a = np.zeros(shape=(legend_height,legend_width))
        for i in range(0,legend_width):
            a[:,i] = self.vmin + (i/legend_width) * (self.vmax-self.vmin)
        save_image(a, self.vmin, self.vmax, path, self.cmap_name)


class LayerWMS:

    def __init__(self, layer_name, layer_label, wms_url, scale, x_coords, y_coords):
        self.layer_name = layer_name
        self.layer_label = layer_label
        self.wms_url = wms_url
        self.cache = {}
        self.failed = set()
        self.scale = scale
        self.x_coords = x_coords
        self.y_coords = y_coords

    def has_legend(self):
        return False

    def check(self, ds):
        for variable in [self.x_coords, self.y_coords]:
            if variable not in ds:
                return f"No variable {variable}"

    def build(self,ds,path,flip=False):
        image_width = ds[self.x_coords].shape[0]
        image_height = ds[self.y_coords].shape[0]
        image_width *= self.scale
        image_height *= self.scale

        xc = ds[self.x_coords]
        yc = ds[self.y_coords]
        if len(xc.shape) != 1:
            raise Exception("x-coords must be 1-dimensional")
        if len(yc.shape) != 1:
            raise Exception("y-coords must be 1-dimensional")
        spacing_x = abs(float(xc[0]) - float(xc[1]))
        spacing_y = abs(float(yc[0]) - float(yc[1]))

        x_min = float(xc.min()) - spacing_x/2
        x_max = float(xc.max()) + spacing_x/2
        y_min = float(yc.min()) - spacing_y/2
        y_max = float(yc.max()) + spacing_y/2
        url = self.wms_url.replace("{WIDTH}",str(image_width)).replace("{HEIGHT}",str(image_height)) \
            .replace("{YMIN}",str(y_min)).replace("{YMAX}",str(y_max)) \
            .replace("{XMIN}",str(x_min)).replace("{XMAX}", str(x_max))

        if url in self.cache:
            os.symlink(self.cache[url],path)
        elif url in self.failed:
            pass
        else:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                self.cache[url] = os.path.split(path)[-1]
            else:
                print(f"Warning - failed to load from {url}")
                self.failed.add(url)

class LayerMask:

    def __init__(self, layer_name, layer_label, band_name, r, g, b, mask):
        self.layer_name = layer_name
        self.layer_label = layer_label
        self.band_name = band_name
        self.r = r
        self.g = g
        self.b = b
        self.mask = mask

    def has_legend(self):
        return False

    def check(self, ds):
        if self.band_name not in ds:
            return f"No variable {self.band_name}"

    def build(self,ds,path,flip):
        da = ds[self.band_name].astype(int)
        if self.mask:
            da = da & self.mask
        arr = da.squeeze().data[:, :]
        if flip:
            arr = np.flipud(arr)
        save_image_mask(arr, path, self.r, self.g, self.b)

class LayerFactory:

    @staticmethod
    def create(layer_name, layer, variable_names):
        layer_type = layer["type"]
        layer_label = layer.get("label", layer_name)
        if layer_type == "single":
            layer_band = layer.get("band", "")
            vmin = layer["min_value"]
            vmax = layer["max_value"]
            cmap = layer.get("cmap", "coolwarm")

            # if the name contains a wildcard, use it to match with one or more variable names in the data
            if "*" in layer_band or "?" in layer_band:
                layers = []
                for variable_name in variable_names:
                    if fnmatch.fnmatch(variable_name, layer_band):
                        layers.append(LayerSingleBand(layer_name, layer_label+"-"+variable_name, variable_name, vmin, vmax, cmap))
                return layers
            else:
                return [LayerSingleBand(layer_name, layer_label, layer_band, vmin, vmax, cmap)]
        elif layer_type == "mask":
            layer_band = layer.get("band", "")
            r = layer["r"]
            g = layer["g"]
            b = layer["b"]
            mask = layer.get("mask", None)
            if "*" in layer_band or "?" in layer_band:
                layers = []
                for variable_name in variable_names:
                    if fnmatch.fnmatch(variable_name, layer_band):
                        layers.append(LayerMask(layer_name, layer_label+"-"+variable_name, variable_name, r, g, b, mask))
                return layers
            else:
             return [LayerMask(layer_name, layer_label, layer_band, r, g, b, mask)]
        elif layer_type == "rgb":
            red_band = layer["red_band"]
            green_band = layer["green_band"]
            blue_band = layer["blue_band"]
            return [LayerRGB(layer_name, layer_label, red_band, green_band, blue_band)]
        elif layer_type == "wms":
            url = layer["url"]
            scale = layer.get("scale", 1)
            x_coords = layer.get("x-coords", "x")
            y_coords = layer.get("y-coords", "y")
            return [LayerWMS(layer_name, layer_label, url, scale, x_coords, y_coords)]
        else:
            raise Exception(f"Unknown layer type {layer_type}")


