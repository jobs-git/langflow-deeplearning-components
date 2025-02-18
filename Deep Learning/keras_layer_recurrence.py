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
from langflow.schema import Data
from langflow.io import IntInput, DropdownInput, BoolInput
import warnings

class KerasRecurrent(Component):
    display_name = "Keras Recurrent Layer"
    description = "Recurrent layer with customizable units, activation, recurrent activation, and return sequences."
    documentation = "https://keras.io/api/layers/recurrent_layers/"
    icon = "layers"
    name = "KerasRecurrent"

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Data",
            required=True,
            info="Input model.",
            input_types=["Sequential"],
        ),
        DropdownInput(
            name="input_recurrent_type",
            display_name="Recurrent Type",
            info="Type of recurrent layer to use.",
            options=["GRU", "LSTM"],
            value="GRU",
            required=True,
        ),
        IntInput(
            name="units",
            display_name="Units",
            info="Number of units in the recurrent layer.",
            value=1,
            required=True,
        ),
        DropdownInput(
            name="activation",
            display_name="Activation",
            info="Activation function to use.",
            options=["tanh", "relu", "sigmoid", "linear"],
            value="tanh",
        ),
        DropdownInput(
            name="recurrent_activation",
            display_name="Recurrent Activation",
            info="Activation function to use for the recurrent step.",
            options=["sigmoid", "relu", "tanh"],
            value="sigmoid",
        ),
        BoolInput(
            name="return_sequences",
            display_name="Return Sequences",
            info="Whether to return the last output in the output sequence, or the full sequence.",
            value=False,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="add_layer"),
    ]

    def add_layer(self) -> Data:
        model = None

        if isinstance(self.input_model, Data):
            model = self.input_model.model
        else:
            raise ValueError("Cannot read input model")

        units = int(self.units)
        activation = self.activation
        recurrent_activation = self.recurrent_activation
        return_sequences = self.return_sequences

        import keras

        if self.input_recurrent_type == "GRU":
            model.add(
                keras.layers.GRU(
                    units=units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    return_sequences=return_sequences
                )
            )
        elif self.input_recurrent_type == "LSTM":
            model.add(
                keras.layers.LSTM(
                    units=units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    return_sequences=return_sequences
                )
            )

        return Data(model=model)
