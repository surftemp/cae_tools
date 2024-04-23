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

import argparse

from cae_tools.utils.data_plotter import DataPlotter


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("data_path",help="path to netcdf4 file containing data")
    parser.add_argument("output_html_path", help="path to write output html to")

    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=True)
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", required=True)


    args = parser.parse_args()

    p = DataPlotter(data_path=args.data_path,
                    input_variables=args.input_variables, output_variable=args.output_variable,
                    output_html_path=args.output_html_path)
    p.run()

if __name__ == '__main__':
    main()