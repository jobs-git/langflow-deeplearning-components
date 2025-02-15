from langflow.custom import Component
from langflow.template import Input, Output
from tensorflow.keras.models import Sequential
from langflow.schema.message import Message
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
            field_type="Message",
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

    def compile_model (self) -> Message:

        model = None
        metrics = None
        loss = None

        if isinstance(self.input_model, Message):
            model = self.input_model.model
        else:
            raise ValueError("Input model should be a Message object containing the model.")

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
        
        return Message(model=model)
