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

import unittest
import os.path

from cae_tools.models.conv_ae_model import ConvAEModel

data_folder = os.path.join(os.path.split(__file__)[0],"..","data")

class QuickTest(unittest.TestCase):

    def test1(self):
        test_name = "16x16_256x256"
        train_path = os.path.join(data_folder, test_name, "train.nc")
        test_path = os.path.join(data_folder,test_name,"test.nc")
        mt = ConvAEModel(fc_size=8,encoded_dim_size=4,nr_epochs=500)
        mt.train("lowres","hires",train_path,test_path)
        mt.save("/tmp/foobar")

        mt2 = ConvAEModel()
        mt2.load("/tmp/foobar")
        mt2.predict(train_path,"lowres","train_scores.nc","hires_estimate")
        mt2.predict(test_path, "lowres","test_scores.nc","hires_estimate")

