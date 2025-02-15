from langflow.custom import Component
from langflow.template import Input, Output
from tensorflow.keras.layers import Conv2D
from langflow.schema.message import Message
from langflow.io import IntInput, DropdownInput, StrInput

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
            value="1, 1",
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
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:

        model = None
        activation = None

        if isinstance(self.input_model, Message):
            model = self.input_model.model
        else:
            raise ValueError("Cannot read input model")

        if self.input_activation != "None":
            activation = self.input_activation

        filters = int(self.filters)

        kernel_size_str = self.kernel_size
        kernel_size = tuple(map(int, kernel_size_str.split(',')))

        model.add (
            tensorflow.keras.layers.Conv2D (
                filters=filters,
                kernel_size=kernel_size,
                activation=activation
            )
        )
        
        return Message(model=model)
