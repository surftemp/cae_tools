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

from htmlfive.html5_builder import ElementFragment


class TableFragment(ElementFragment):

    def __init__(self, attrs={},style={}):
        super().__init__("table",attrs,style)

    def add_row(self, cells):
        tr = self.add_element("tr")
        for cell in cells:
            td = tr.add_element("td")
            if isinstance(cell,str):
                td.add_text(cell)
            else:
                td.add_fragment(cell)

