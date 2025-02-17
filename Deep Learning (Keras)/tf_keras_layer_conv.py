"""
Copyright (c) 2025 Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Author: James Guana
"""

from langflow.custom import Component
from langflow.template import Input, Output
from tensorflow.keras.layers import Conv2D
from langflow.schema.message import Message
from langflow.io import IntInput, DropdownInput, StrInput
from tensorflow.keras.models import Sequential
import re

class KerasConv (Component):
    display_name = "Keras Conv Layer"
    description = "Convolutional layer with customizable filters, kernel size, and activation."
    documentation = "https://keras.io/api/layers/convolution_layers/"
    icon = "layers"
    name = "KerasConv"

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Message",
            required=True,
            info="Input model.",
            input_types=["Sequential"],
        ),
        DropdownInput(
            name="input_conv_type",
            display_name="Convolution Type",
            info="Type of convolutional layer to use.",
            options=["Conv1D", "Conv2D", "Conv3D"],
            value="Conv1D",
            required=True,
        ),
        IntInput(
            name="filters",
            display_name="Filters",
            info="Number of filters in the Conv layer.",
            value=1,
            required=True,
        ),
        StrInput(
            name="kernel_size",
            display_name="Kernel Size",
            info="Size of the kernel as a comma-separated list, e.g., 3, 3.",
            value="1",
        ),
        DropdownInput(
            name="input_activation",
            display_name="Activation",
            info="Activation function to use.",
            options=["None", "relu", "sigmoid", "tanh", "softmax"],
            value="None",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="add_layer"),
    ]

    def validate_input_shape(self, input_shape: str) -> bool:
        pattern = re.compile(r'^(\d+\s*,\s*)*\d+$')
        return bool(pattern.match(input_shape.strip()))

    def add_layer(self) -> Message:
        model = None
        activation = None

        if isinstance(self.input_model, Message):
            model = self.input_model.model
        else:
            raise ValueError("Cannot read input model")

        kernel_size_str = self.kernel_size.replace(" ", "")  # Remove any spaces

        if not self.validate_input_shape(kernel_size_str):
            raise ValueError("Kernel size should contain only numbers and commas.")

        kernel_size = tuple(map(int, kernel_size_str.split(',')))

        required_length = {"Conv1D": 1, "Conv2D": 2, "Conv3D": 3}[self.input_conv_type]

        if len(kernel_size) < required_length:
            raise ValueError(f"{self.input_conv_type} requires at least {required_length} values for kernel size.")
        elif len(kernel_size) > required_length:
            warnings.warn(f"{self.input_conv_type} only requires {required_length} values for kernel size. Using the first {required_length} values.")
            kernel_size = kernel_size[:required_length]

        if self.input_activation != "None":
            activation = self.input_activation

        filters = int(self.filters)

        if self.input_conv_type == "Conv1D":
            model.add(
                tensorflow.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size[0],  # Use the first value
                    activation=activation
                )
            )
        elif self.input_conv_type == "Conv2D":
            model.add(
                tensorflow.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,  # Use the first two values
                    activation=activation
                )
            )
        elif self.input_conv_type == "Conv3D":
            model.add(
                tensorflow.keras.layers.Conv3D(
                    filters=filters,
                    kernel_size=kernel_size,  # Use the first three values
                    activation=activation
                )
            )

        return Message(model=model)