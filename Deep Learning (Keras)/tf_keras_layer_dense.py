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
from tensorflow.keras.layers import InputLayer
from langflow.schema.message import Message
import tensorflow as tf
from tensorflow.keras.models import Sequential
from langflow.io import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    SecretStrInput,
    StrInput,
)
import re

class KerasDense (Component):
    display_name = "Keras Dense"
    description = "Adds a dense layer with the specified units to the Keras model."
    documentation = "https://keras.io/api/layers/core_layers/dense/"
    icon = "layers"
    name = "KerasDense"

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Message",
            info="Input model.",
            required=True,
            input_types=["Sequential"]
        ),
        IntInput(
            name="input_units",
            display_name="Units",
            info="Number of units in the Dense layer.",
            value=1,
            required=True,
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
        Output (display_name = "Output", name = "output", method = "add_layer"),
    ]

    def add_layer (self) -> Message:

        if isinstance (self.input_model, Message):
            model = self.input_model.model
        else:
            raise ValueError("Sequential model not initialized.")

        activation = None

        if self.input_activation != "None":
            activation = self.input_activation

        model.add (
            tf.keras.layers.Dense (
                units = self.input_units, 
                activation = activation
            )
        )

        return Message (model = model)
