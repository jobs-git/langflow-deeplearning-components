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
from langflow.schema import Data, DataFrame
from langflow.io import DataFrameInput, IntInput
import numpy as np

class KerasFit(Component):
    display_name = "Keras Fit"
    description = "Fits the Keras model on the provided data."
    documentation = "https://keras.io/api/models/model_training_apis/#fit-method"
    icon = "train"
    name = "KerasFit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = None

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Data",
            required=True,
            info="Input model to be trained.",
            input_types=["Sequential"],
        ),
        DataFrameInput(
            name="x",
            display_name="Input Data (x)",
            info="Input data for training.",
            field_type="DataFrame",
            required=True,
        ),
        DataFrameInput(
            name="y",
            display_name="Target Data (y)",
            info="Target data for training.",
            field_type="DataFrame",
        ),
        IntInput(
            name="input_epochs",
            display_name="Epochs",
            info="Number of backpropagation.",
            value=1,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Number of samples per gradient update.",
            value=None,
        ),
    ]

    outputs = [
        Output(display_name="Model", name="model", method="fit_model"),
        Output(display_name="History", name="history", method="get_history")
    ]

    def fit_model(self) -> Data:
        model = None

        if isinstance(self.input_model, Data):
            model = self.input_model.model
        else:
            raise ValueError("Cannot read input model")
        
        x = self.x.squeeze(axis=1).values.astype(np.float32)
        y = self.y.squeeze(axis=1).values.astype(np.float32)
        
        batch_size = self.batch_size if self.batch_size is not None else 32

        self.history = model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=self.input_epochs
        )
        
        return Data(model=model)
    
    def get_history (self) -> Data:

        self.fit_model()

        if self.history is None:
            raise ValueError("Cannot retrieve training history")

        return Data (history=self.history)