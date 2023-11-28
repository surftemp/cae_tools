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


class DataGenerator:

    """Generate Test Data Patterns for ConvAEModel"""

    def __init__(self, input_size, output_size, pattern="circle"):
        self.input_size = input_size
        self.output_size = output_size
        self.pattern = pattern

    def gen(self,height,width,mu=1.0):
        if self.pattern == "circle":
            x, y = np.meshgrid(np.linspace(-3,3,height), np.linspace(-2,2,width))
            d = np.sqrt(x*x+y*y)
            sigma = 0.2
            g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            g = ndimage.rotate(g,15)[0:height,0:width]
            return g
        elif self.pattern == "curve":
            x, y = np.meshgrid(np.linspace(0, 100, width),np.linspace(0, 100, height))
            cx = 50
            cy = 50
            max_d = math.sqrt(50**2+50**2)
            g = np.sqrt((x-cx)**2+(y-cy)**2)/max_d
            return g

    def generate_data(self,n):
        arr1 = np.zeros((n,1,self.input_size[0],self.input_size[1]),dtype=np.float32)
        noise = [random.random() for i in range(n)]

        for i in range(n):
            arr1[i,0,:,:] = 288+self.gen(self.input_size[0],self.input_size[1])*noise[i]*5

        da1 = xr.DataArray(data=arr1,dims=("n","chan","y1","x1"))

        arr2 = np.zeros((n, 1, self.output_size[0], self.output_size[1]), dtype=np.float32)
        for i in range(n):
            arr2[i, 0, :, :] = 288 + self.gen(self.output_size[0],self.output_size[1]) * noise[i] * 5

        da2 = xr.DataArray(data=arr2, dims=("n", "chan", "y2", "x2"))

        return (da1,da2)


def main():
    import os

    data_root_folder = os.path.join(os.path.split(__file__)[0],"..","data")

    for test_spec in [((16,16),(256,256),"circle"),((16,16),(256,256),"curve"),((24,20),(280,256),"circle")]:

        ((i_h,i_w),(o_h,o_w), pattern) = test_spec
        folder = os.path.join(data_root_folder,pattern,f"{i_h}x{i_w}_{o_h}x{o_w}")

        if not os.path.exists(folder):
            print("Generating test data:" + str(test_spec))
            os.makedirs(folder)

            for filename in ["train.nc","test.nc"]:

                dg = DataGenerator((i_h,i_w),(o_h,o_w),pattern)
                (input_da,output_da) = dg.generate_data(100)

                ds = xr.Dataset()
                ds["hires"] = output_da
                ds["lowres"] = input_da
                path = os.path.join(folder,filename)
                ds.to_netcdf(path)
                print(f"Written {path}")

if __name__ == '__main__':
    main()


