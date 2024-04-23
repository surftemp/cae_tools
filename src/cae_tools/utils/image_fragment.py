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

from .html5.html5_builder import Html5Builder, Fragment, ElementFragment
import base64

from .utils import add_exo_dependencies, prepare_attrs


class ImageFragment(ElementFragment):

    def __init__(self, src, alt_text="", w=None, h=None):
        super().__init__("img", prepare_attrs({
            "src": src, "alt":alt_text, "width": w, "height": h}))


def inlined_image(from_path):
    if from_path.endswith("gif"):
        mime_type = "image/gif"
    elif from_path.endswith("png"):
        mime_type = "image/png"
    elif from_path.endswith("jpeg") or from_path.endswith("jpg"):
        mime_type = "image/jpeg"
    else:
        raise Exception("Unable to guess mime type for: "+from_path)
    with open(from_path,"rb") as f:
        content_bytes = f.read()
    return "data:" + mime_type + ";charset=US-ASCII;base64," + str(base64.b64encode(content_bytes), "utf-8")


class InlineImageFragment(ElementFragment):

    def __init__(self, path, alt_text="", w=None, h=None):
        super().__init__("img", prepare_attrs({
            "src": inlined_image(path), "alt":alt_text, "width": w, "height": h}))

