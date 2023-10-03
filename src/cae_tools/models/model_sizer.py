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

class LayerSpec:

    def __init__(self, is_input=True, kernel_size=3, stride=2, input_dimensions=None, output_dimensions=None, output_padding_y=0,
                 output_padding_x=0):
        self.is_input = is_input
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.output_padding_y = output_padding_y
        self.output_padding_x = output_padding_x

    def __repr__(self):
        s = "\tInputLayer:\n" if self.is_input else "\tOutputLayer:\n"
        s += f"\t\tkernel_size={self.kernel_size}  stride={self.stride} output_padding=({self.output_padding_y},{self.output_padding_x})\n"
        s += f"\t\t{self.input_dimensions} => {self.output_dimensions}\n"
        return s

    def get_kernel_size(self):
        return self.kernel_size

    def get_stride(self):
        return self.stride

    def get_input_dimensions(self):
        return self.input_dimensions

    def get_output_dimensions(self):
        return self.output_dimensions

    def get_output_padding(self):
        return (self.output_padding_y, self.output_padding_x)

    def save(self):
        return {
            "is_input": self.is_input,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "output_padding_x": self.output_padding_x,
            "output_padding_y": self.output_padding_y,
            "input_dimensions": list(self.input_dimensions),
            "output_dimensions": list(self.output_dimensions)
        }

    def load(self, from_obj):
        self.is_input = from_obj["is_input"]
        self.kernel_size = from_obj["kernel_size"]
        self.stride = from_obj["stride"]
        self.output_padding_x = from_obj["output_padding_x"]
        self.output_padding_y = from_obj["output_padding_y"]
        self.input_dimensions = tuple(from_obj["input_dimensions"])
        self.output_dimensions = tuple(from_obj["output_dimensions"])


class ModelSpec:

    def __init__(self, input_layer_specs=[], output_layer_specs=[]):
        self.input_layers = input_layer_specs
        self.output_layers = output_layer_specs

    def get_input_layers(self):
        return self.input_layers

    def get_output_layers(self):
        return self.output_layers

    def __repr__(self):
        s = "ModelSpec:\n"
        for input_spec in self.input_layers:
            s += str(input_spec)
        for output_spec in self.output_layers:
            s += str(output_spec)
        return s

    def save(self):
        return {
            "input_layers": list(map(lambda l: l.save(), self.input_layers)),
            "output_layers": list(map(lambda l: l.save(), self.output_layers)),
        }

    def load(self,from_obj):
        self.input_layers = []
        self.output_layers = []
        input_layers = from_obj["input_layers"]
        for idx in range(len(input_layers)):
            layer_spec = LayerSpec()
            layer_spec.load(input_layers[idx])
            self.input_layers.append(layer_spec)
        output_layers = from_obj["output_layers"]
        for idx in range(len(output_layers)):
            layer_spec = LayerSpec()
            layer_spec.load(output_layers[idx])
            self.output_layers.append(layer_spec)


def create_model_spec(input_size=(7, 7), input_channels=1, output_size=(28, 28), output_channels=1, stride=2,
                      kernel_size=3, limit=3):
    size = input_size
    channels = input_channels
    input_layers = []

    while min(size) > limit:
        (size_y, size_x) = size
        input_dims = (int(channels), int(size_y), int(size_x))
        size_x = ((size_x - (kernel_size - 1) - 1) // stride) + 1
        size_y = ((size_y - (kernel_size - 1) - 1) // stride) + 1
        channels *= 2
        output_dims = (int(channels), int(size_y), int(size_x))
        input_layers.append(LayerSpec(True, kernel_size, stride, input_dims, output_dims))
        size = (size_y, size_x)

    output_layers = []

    size = output_size
    channels = output_channels
    while min(size) > limit:
        (size_y, size_x) = size
        output_dims = (int(channels), int(size_y), int(size_x))
        input_size_x = ((size_x - (kernel_size - 1) - 1) // stride) + 1
        input_size_y = ((size_y - (kernel_size - 1) - 1) // stride) + 1
        output_size_x = (input_size_x - 1) * stride + (kernel_size - 1) + 1
        output_size_y = (input_size_y - 1) * stride + (kernel_size - 1) + 1
        output_padding_x = size_x - output_size_x
        output_padding_y = size_y - output_size_y
        channels *= 2
        input_dims = (int(channels), int(input_size_y), int(input_size_x))
        output_layers.insert(0, LayerSpec(False, kernel_size, stride, input_dims, output_dims,
                                          output_padding_y=output_padding_y, output_padding_x=output_padding_x))
        size = (input_size_y, input_size_x)

    spec = ModelSpec(input_layers, output_layers)
    return spec

