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
import math

import test_specs

class DataGenerator:

    """Generate Test Data Patterns for ConvAEModel"""

    def __init__(self, input_size, output_size, pattern="circle"):
        self.input_size = input_size
        self.output_size = output_size
        self.pattern = pattern
        self.aux_data = {}
        self.aux_data_range = {}
        if pattern == "tidal_circle":
            self.aux_data_range["tide"] = (-1.0, 1.0)
        self.n = 0

    @staticmethod
    def lcm(n0, n1):
        v0 = n0
        v1 = n1
        while True:
            if v0 == v1:
                return v0
            elif v0 < v1:
                v0 += n0
            else:
                v1 += n1

    def gen(self,index,height,width,mu=1.0):
        if self.pattern == "circle":
            y, x = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-3, 3, height))
            d = np.sqrt(y * y + x * x)
            sigma = 0.2
            g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            return ndimage.rotate(g, 15)[0:height, 0:width]
        elif self.pattern == "tidal_circle":
            if "tide" not in self.aux_data:
                self.aux_data["tide"] = np.zeros(shape=(self.n,),dtype=np.float32)
            tidal_height = math.sin(random.random() * 2*math.pi)
            self.aux_data["tide"][index] = tidal_height
            y, x = np.meshgrid(np.linspace(-8,8,width),np.linspace(-10,10,height))
            d = np.sqrt(y*y+x*x)
            sigma = 0.2 + 0.1*tidal_height
            g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            return ndimage.rotate(g,15)[0:height,0:width]
        elif self.pattern == "curve":
            y, x = np.meshgrid(np.linspace(0, 100, width),np.linspace(0,100,height))
            cx = 50
            cy = 50
            max_d = math.sqrt(50**2+50**2)
            g = np.sqrt((y-cy)**2+(x-cx)**2)/max_d
            return g

    def generate_data(self,n):
        self.n = n
        self.aux_data = {}
        noise = [random.random() for i in range(n)]

        sample_height = DataGenerator.lcm(self.output_size[0],self.input_size[0])
        sample_width = DataGenerator.lcm(self.output_size[1], self.input_size[1])

        sample_size = (sample_height,sample_width)

        input_arr = np.zeros((n,1, self.input_size[0],self.input_size[1]),dtype=np.float32)
        output_arr = np.zeros((n, 1, self.output_size[0], self.output_size[1]),dtype=np.float32)

        for i in range(n):
            arr = 288 + self.gen(i,sample_size[0], sample_size[1]) * noise[i] * 5
            das = xr.DataArray(data=arr, dims=("ys", "xs"))
            input_arr[i,0,:,:] = das.coarsen({"ys":sample_height//self.input_size[0], "xs":sample_width//self.input_size[1]}).mean().values
            output_arr[i,0,:,:] = das.coarsen({"ys":sample_height//self.output_size[0], "xs": sample_width//self.output_size[1]}).mean().values
            print(str(100*(i/n))+"% complete")

        da1 = xr.DataArray(data=input_arr,dims=("n","chan","y1","x1")) # input
        da2 = xr.DataArray(data=output_arr, dims=("n", "chan", "y2", "x2")) #output
        aux_das = {}
        for key in self.aux_data:
            (range_min, range_max) = self.aux_data_range[key]
            aux_das[key] = xr.DataArray(self.aux_data[key], dims=("n",),
                                        attrs={"type": "auxilary-predictor",
                                               "min-value": range_min, "max-value": range_max})

        return (da1,da2,aux_das)


def main():
    import os

    n = 100
    data_root_folder = os.path.join(os.path.split(__file__)[0],"..","data")

    for test_spec_name in test_specs.all_specs:

        test_spec = test_specs.all_specs[test_spec_name]
        (i_h, i_w) = test_spec["input_size"]
        (o_h,o_w) = test_spec["output_size"]
        pattern = test_spec["pattern"]
        input_names = test_spec["inputs"]
        output_name = test_spec["output"]

        folder = os.path.join(data_root_folder,test_spec_name,f"{i_h}x{i_w}_{o_h}x{o_w}")

        already_generated = True
        for filename in ["train.nc", "test.nc"]:
            if not os.path.exists(os.path.join(folder,filename)):
                already_generated = False

        if not already_generated:
            print("Generating test data:" + str(test_spec))
            os.makedirs(folder, exist_ok=True)

            for filename in ["train.nc","test.nc"]:

                dg = DataGenerator((i_h,i_w),(o_h,o_w),pattern)
                (input_da,output_da,aux_das) = dg.generate_data(n)

                ds = xr.Dataset()
                ds[output_name] = output_da
                ds[input_names[0]] = input_da
                input_idx = 1
                for key in aux_das:
                    ds[key+"_1d"] = aux_das[key]
                    arr = np.zeros((n,1,i_h,i_w),dtype=np.float32)
                    arr[:,:,:,:] = np.reshape(aux_das[key].data,newshape=(n,1,1,1))
                    ds[input_names[input_idx]] = xr.DataArray(arr,dims=("n","chan","y1","x1"),attrs={})
                    input_idx += 1
                path = os.path.join(folder,filename)
                ds.to_netcdf(path)
                print(f"Written {path}")

if __name__ == '__main__':
    main()


