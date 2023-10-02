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



import numpy as np
from scipy import ndimage
import xarray as xr
import random


class DataGenerator:

    """Generate Test Data Patterns for ConvAEModel"""

    def __init__(self, target_size, coarsen_by):
        self.target_size = target_size
        self.coarsen_by = coarsen_by

    def gen(self,mu=1.0):
        x, y = np.meshgrid(np.linspace(-3,3,self.target_size), np.linspace(-1,2,self.target_size))
        d = np.sqrt(x*x+y*y)
        sigma = 0.2
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        g = ndimage.rotate(g,15)[0:self.target_size,0:self.target_size]
        return g

    def generate_data(self,n):
        arr = np.zeros((n,1,256,256),dtype=np.float32)
        for i in range(n):
            arr[i,0,:,:] = 288+self.gen()*random.random()*5

        da = xr.DataArray(data=arr,dims=("n","chan","y","x"))
        return (da,self.coarsen(da))

    def coarsen(self,da):
        return xr.DataArray(da.coarsen({"x":self.coarsen_by,"y":self.coarsen_by}).mean().values,dims=("n","chan","y2","x2"))

import os

data_root_folder = os.path.join(os.path.split(__file__)[0],"..","data")

folder = os.path.join(data_root_folder,"16x16_256x256")

os.makedirs(folder,exist_ok=True)

for filename in ["train.nc","test.nc"]:

    dg = DataGenerator(256,16)
    (d,d2) = dg.generate_data(100)

    ds = xr.Dataset()
    ds["hires"] = d
    ds["lowres"] = d2
    path = os.path.join(folder,filename)
    ds.to_netcdf(path)
    print(f"Written {path}")




