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

import io
import html as htmlutils
import xml.dom.minidom
from .html5_common import HTML5_DOCTYPE, require_end_tags, void_elements


class Html5Exporter:
    """
    Export an XML dom as html5

    Args
        indent_spaces: number of spaces to make up each indent

    A way you might use me is:

    >>> from htmlfive import Html5Exporter
    >>> from xml.dom.minidom import getDOMImplementation
    >>> doc = getDomImplementation().createDocument(None, "html", None)
    >>> body = doc.createElement("body")
    >>> doc.documentElement.appendChild(body)
    >>> txt = doc.createTextNode("Hello")
    >>> body.appendChild(txt)
    >>> exporter = Html5Exporter()
    >>> html = exporter.export(doc)
    >>> print(html)
    <!DOCTYPE html>
    <html>
        <body>
            Hello
        </body>
    </html>
    """

    def __init__(self, indent_spaces: int = 4):
        self.indent_spaces = indent_spaces

    def __is_ws(self, txt):
        txt = txt.replace(" ", "").replace("\t", "").replace("\n", "")
        return txt == ""

    def __exportElement(self, ele, indent):
        self.of.write(indent * " " * self.indent_spaces)
        self.of.write("<" + ele.tagName)
        attr_count = 0
        for (k, v) in ele.attributes.items():

            if v is None:
                self.of.write(" %s" % k)
            else:
                if not isinstance(v, str):
                    v = str(v)
                if '"' in v:
                    if "'" in v:
                        self.of.write(' %s="%s"' % (k, htmlutils.escape(v)))
                    else:
                        self.of.write(" %s='%s'" % (
                            k, htmlutils.escape(v, quote=False)))  # single quote values containing double quote
                else:
                    self.of.write(' %s="%s"' % (k, htmlutils.escape(v, quote=False)))
            attr_count += 1
        child_count = len(ele.childNodes)

        if ele.tagName in require_end_tags or child_count > 0:
            self.of.write(">")
            if child_count:
                self.of.write("\n")
                for childNode in ele.childNodes:
                    if childNode.nodeType == childNode.ELEMENT_NODE:
                        self.__exportElement(childNode, indent + 1)
                    elif childNode.nodeType == childNode.TEXT_NODE:
                        self.__exportText(childNode, indent + 1)
                    elif childNode.nodeType == childNode.COMMENT_NODE:
                        self.__exportComment(childNode, indent + 1)
                self.of.write(" " * indent * self.indent_spaces + "</%s>" % ele.tagName)
            else:
                self.of.write("</%s>" % ele.tagName)
        else:
            if ele.tagName not in void_elements:
                self.of.write("/>")
            else:
                self.of.write(">")
        self.of.write("\n")

    def __exportText(self, tn, indent):
        txt = tn.data.rstrip(" \n").lstrip(" \n")
        if not self.__is_ws(txt):
            self.of.write(" " * indent * self.indent_spaces)
            # self.of.write(htmlutils.escape(txt))
            self.of.write(txt)
            self.of.write("\n")

    def __exportComment(self, cn, indent):
        txt = cn.data.rstrip(" \n").lstrip(" \n")
        self.of.write(" " * indent * self.indent_spaces)
        self.of.write("<!--")
        self.of.write(txt)
        self.of.write("-->")
        self.of.write("\n")

    def export(self, doc: xml.dom.minidom.Document) -> str:
        """
        Export a DOM to an HTML string.

        Args:
            doc: the DOM document to export

        Returns:
            A string containing the HTML
        """
        with io.StringIO() as self.of:
            self.of.write(HTML5_DOCTYPE + "\n")
            ele = doc.documentElement
            self.__exportElement(ele, 0)
            return self.of.getvalue()
