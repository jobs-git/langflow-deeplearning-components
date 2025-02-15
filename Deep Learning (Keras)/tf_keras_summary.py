from langflow.custom import Component
from langflow.template import Input, Output
from langflow.schema.message import Message
import io
import contextlib
import tensorflow as tf

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
            field_type="Message",
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

        if isinstance(self.input_model, Message):
            model = self.input_model.model

            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                model.summary()
                model_summary = buf.getvalue()

        self.reset_keras(model)

        return model_summary

    def reset_keras(self, model):
        del model
        gc.collect ()
        a = tf.zeros([], tf.float32)
        del a
        gc.collect ()
        tf.keras.backend.clear_session()
