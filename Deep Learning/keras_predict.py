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
from langflow.schema import DataFrame

class KerasPredict(Component):
    display_name = "Keras Predict"
    description = "Predicts the vector of the provided data."
    documentation = "https://keras.io/api/models/model_training_apis/#predict-method"
    icon = "train"
    name = "KerasPredict"

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
            display_name="Data (x)",
            info="Target data for training.",
            field_type="DataFrame",
        ),
    ]

    outputs = [
        Output(display_name="Predictions", name="predict", method="predict")
    ]

    def predict (self) -> DataFrame:
        model = None

        if isinstance(self.input_model, Data):
            model = self.input_model.model
        else:
            raise ValueError("Cannot read input model")

        x = self.x.squeeze(axis=1).values.astype(np.float32)

        results = model.predict(
            x=x
        )

        data_dict = {"y_pred": results.tolist()}
        data = DataFrame(data_dict)
        
        return data