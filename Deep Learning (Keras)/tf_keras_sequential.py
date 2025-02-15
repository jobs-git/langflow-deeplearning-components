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

import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"

class KerasSequential (Component):
    display_name = "Keras Sequential"
    description = "Creates a sequential model."
    documentation = "https://keras.io/api/models/sequential/"
    icon = "input"
    name = "KerasSequential"

    outputs = [
        Output(display_name="Output", name="output", method="create_sequential"),
    ]

    def create_sequential (self) -> Message:

        model = Sequential()

        return Message(model=model)
