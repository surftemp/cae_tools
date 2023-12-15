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

def prepare_attrs(d):
    return {k:str(v) for (k,v) in d.items() if v is not None}

def add_exo_dependencies(head_fragment,exo_version="latest"):

    head_fragment.add_element("link",{"href":"//fonts.googleapis.com/css?family=Raleway:400,300,600",
                                 "rel":"stylesheet", "type":"text/css"})
    head_fragment.add_element("link", {"href": "https://visual-topology.github.io/exo/versions/%s/exo.css"%str(exo_version),
                                  "rel": "stylesheet", "type": "text/css"})
    head_fragment.add_element("link", {"href": "https://visual-topology.github.io/exo/versions/%s/exo-icons.css" % str(exo_version),
                                  "rel": "stylesheet", "type": "text/css"})
    head_fragment.add_element("script", {"src":"https://visual-topology.github.io/exo/versions/%s/exo.js"%str(exo_version),
                                                                    "type":"text/javascript"})

anti_aliasing_style = """
img { 
    image-rendering: optimizeSpeed;             /* STOP SMOOTHING, GIVE ME SPEED  */
    image-rendering: -moz-crisp-edges;          /* Firefox                        */
    image-rendering: -o-crisp-edges;            /* Opera                          */
    image-rendering: -webkit-optimize-contrast; /* Chrome (and eventually Safari) */
    image-rendering: pixelated;                 /* Universal support since 2021   */
    image-rendering: optimize-contrast;         /* CSS3 Proposed                  */
    -ms-interpolation-mode: nearest-neighbor;   /* IE8+                           */
}"""


