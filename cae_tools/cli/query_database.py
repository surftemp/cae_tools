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

from cae_tools.utils.model_database import ModelDatabase
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database_path")
    parser.add_argument("--model-id", type=str, help="Dump details for this specific model", default=None)
    args = parser.parse_args()
    md = ModelDatabase(args.database_path)
    if args.model_id:
        md.dump_model(model_id=args.model_id)
    else:
        md.dump()

if __name__ == '__main__':
    main()