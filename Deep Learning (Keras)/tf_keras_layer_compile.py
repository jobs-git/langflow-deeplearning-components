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
from tensorflow.keras.models import Sequential
from langflow.schema import Data
from langflow.io import StrInput, DropdownInput
from tensorflow.keras.models import Sequential

class KerasCompile (Component):
    display_name = "Keras Compile"
    description = "Compiles the Keras model with specified optimizer, loss, and metrics."
    documentation = "https://keras.io/api/models/model_training_apis/#compile-method"
    icon = "compile"
    name = "KerasCompile"

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Data",
            info="Input model.",
            required=True,
            input_types=["Sequential"],
        ),
        DropdownInput(
            name="optimizer",
            display_name="Optimizer",
            info="Select the optimizer for model compilation.",
            options=[
                "rmsprop"
                "adam",
                "sgd",
                "rmsprop",
                "adagrad",
                "adadelta",
            ],
            value="rmsprop",
        ),
        DropdownInput(
            name="input_loss",
            display_name="Loss",
            info="Select the loss function for model compilation.",
            options=[
                "None",
                "sparse_categorical_crossentropy",
                "categorical_crossentropy",
                "binary_crossentropy",
                "mean_squared_error",
                "mean_absolute_error",
            ],
            value="None",
        ),
        DropdownInput(
            name="input_metrics",
            display_name="Metrics",
            info="Select the metrics for model evaluation.",
            options=[
                "None",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
            ],
            value="None",
        ),
    ]

    outputs = [
        Output(display_name="Compiled Model", name="output_model", method="compile_model"),
    ]

    def compile_model (self) -> Data:

        model = None
        metrics = None
        loss = None

        if isinstance(self.input_model, Data):
            model = self.input_model.model
        else:
            raise ValueError("Input model should be a Data object containing the model.")

        if self.input_metrics != "None":
            metrics = self.input_metrics

        if self.input_loss != None:
            loss = self.input_loss

        optimizer = self.optimizer  

        # Compile the model
        model.compile(
            optimizer=optimizer, 
            loss=loss, 
            metrics=metrics
        )
        
        return Data(model=model)
