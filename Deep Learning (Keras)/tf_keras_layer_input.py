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

class KerasInput (Component):
    display_name = "Keras Input"
    description = "Adds an input layer with the specified shape to the Keras model."
    documentation = "https://keras.io/api/layers/core_layers/input/"
    icon = "layers"
    name = "KerasInput"

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Message",
            info="Input model.",
            required=True,
            input_types=["Sequential"]
        ),
        Input(
            name="input_shape",
            display_name="Data Shape",
            info="Input shape as a comma-separated list, e.g., 28, 28, 1 or 1.",
            required=True,
            value="1",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="add_input_layer"),
    ]

    def validate_input_shape (self, input_shape: str) -> bool:
 
        pattern = re.compile(r'^(\d+\s*,\s*)*\d+$')
        return bool(pattern.match(input_shape.strip()))

    def add_input_layer(self) -> Message:


        if isinstance(self.input_model, Message):
            model = self.input_model.model
        else:
            raise ValueError("Sequential model not initialized.")

        input_shape_str = self.input_shape.replace(" ", "")

        if not self.validate_input_shape(input_shape_str):
            raise ValueError("Invalid input shape. Only numbers and commas are allowed.")

        input_shape = tuple(map(int, input_shape_str.split(',')))

        model.add(
            InputLayer(
                input_shape=input_shape
            )
        )

        return Message(model=model)
