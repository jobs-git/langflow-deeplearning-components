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
import io
import contextlib
import gc

class KerasSummary(Component):
    display_name = "Keras Summary"
    description = "Prints the summary of a Keras model."
    documentation = "https://keras.io/api/models/model/"
    icon = "summary"
    name = "KerasSummary"

    inputs = [
        Input(
            name="input_model",
            display_name="Model",
            field_type="Data",
            info="Input model.",
            input_types=["Sequential"]
        ),
    ]

    outputs = [
        Output(display_name="Model Summary", name="summary", method="print_summary"),
    ]

    def print_summary(self) -> str:
        model_summary = ""
        model = None

        if isinstance(self.input_model, Data):
            model = self.input_model.model

            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                model.summary()
                model_summary = buf.getvalue()

        self.reset_keras(model)

        return model_summary

    def reset_keras(self, model):
        del model
        gc.collect ()
        # a = tf.zeros([], tf.float32)
        # del a
        # gc.collect ()
        # tf.keras.backend.clear_session()
