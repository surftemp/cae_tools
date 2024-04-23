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
import xml.dom.minidom
from xml.dom.minidom import getDOMImplementation
import typing
from .html5_exporter import Html5Exporter


class Fragment:

    def get_node(self, builder:"Html5Builder"):
        pass


class TextFragment(Fragment):
    """
    Represent an HTML5 text node.
    """

    def __init__(self, text):
        self.text = text

    def get_node(self, builder):
        return builder.doc.createTextNode(self.text)


class ElementFragment(Fragment):
    """
    Represent a generic HTML5 element node.
    """

    def __init__(self, tag: str, attrs: typing.Dict[str, str] = {},
                 style: typing.Dict[str, str] = {}):
        self.tag = tag
        self.attrs = attrs
        self.style = style
        self.child_fragments = []


    def add_element(self, tag: str, attrs: typing.Dict[str, str] = {},
                    style: typing.Dict[str, str] = {}) -> "ElementFragment":
        """
        Add a child element fragment to this fragment

        Arguments:
            tag: the tag name to add
            attrs: dictionary containing the names and values of attributes.
            style: dictionary contiaining the names and values of CSS styles to apply.

        Returns:
             The child fragment that was added
        """
        fragment = ElementFragment(tag, attrs, style)
        self.add_fragment(fragment)
        return fragment

    def add_text(self, text: str) -> "ElementFragment":
        """
        Add a child text fragment to this fragment

        Arguments:
            text: the text to include
        """
        fragment = TextFragment(text)
        self.add_fragment(fragment)
        return self

    def set_attribute(self,name,value) -> "ElementFragment":
        self.attrs[name] = value
        return self

    def set_style(self, name, value) -> "ElementFragment":
        self.style[name] = value
        return self

    def add_fragment(self, fragment: Fragment) -> "Html5Builder":
        self.child_fragments.append(fragment)
        return self

    def get_node(self, builder: "Html5Builder") -> xml.dom.minidom.Node:
        node = builder.doc.createElement(self.tag)
        for (name, value) in self.attrs.items():
            node.setAttribute(name, value)
        if self.style:
            style_value = ""
            for (name, value) in self.style.items():
                style_value += name + ":" + str(value) + ";"
            node.setAttribute("style", style_value)
        for fragment in self.child_fragments:
            node.appendChild(fragment.get_node(builder))
        return node


class RawFragment:

    def __init__(self, node):
        self.node = node

    def get_node(self, builder):
        return self.node

class Html5Builder:
    """
    Create and populate an html5 document

    A way you might use me is:

    >>> builder = Html5Builder(language="en")
    >>> builder.head().add_element("title").add_text("Title!")
    >>> builder.body().add_element("h1",{"id":"heading"}).add_text("Heading")
    >>> builder.body().add_element("div").add_text("Lorem Ipsum")
    >>> print(builder.get_html())
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <title>
                Title!
            </title>
        </head>
        <body>
            <h1 id="heading">
                Heading
            </h1>
            <div>
                Lorem Ipsum
            </div>
        </body>
    </html>
    """

    def __init__(self, language: str = "", id_suffix="_bld", width=None):
        self.doc = getDOMImplementation().createDocument(None, "html", None)
        self.root = self.doc.documentElement
        if language:
            self.root.setAttribute("lang", language)
        self.__head = ElementFragment("head")
        self.__body = ElementFragment("body")
        self.css = ""
        if width:
            self.css = "body { margin-left:auto; margin-right:auto; position:relative; width:%dpx; }"%(width)
        self.post_build_fns = []
        self.id_counters = {}
        self.id_suffix = id_suffix

    def head(self) -> ElementFragment:
        """
        Get the head fragment of the document being built

        Returns:
             Html <head> fragment
        """
        return self.__head

    def body(self) -> ElementFragment:
        """
        Get the body fragment of the document being built

        Returns:
             Html <body> fragment
        """
        return self.__body

    def get_html(self) -> str:
        """
        Get an HTML5 string representation of the document being built

        Returns:
             Html formatted string
        """
        exporter = Html5Exporter()
        if self.css:
            self.__head.add_element("style").add_text(self.css)
        head_node = self.__head.get_node(self)
        body_node = self.__body.get_node(self)

        for fn in self.post_build_fns:
            fn(head_node,body_node)

        self.root.appendChild(head_node)
        self.root.appendChild(body_node)
        return exporter.export(self.doc).strip()

    def register_post_build(self,fn):
        self.post_build_fns.append(fn)

    def get_next_id(self,prefix):
        if prefix not in self.id_counters:
            self.id_counters[prefix] = 0
        id = prefix+str(self.id_counters[prefix])+self.id_suffix
        self.id_counters[prefix] =  1 + self.id_counters[prefix]
        return id



