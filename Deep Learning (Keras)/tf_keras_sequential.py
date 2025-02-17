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
from tensorflow.keras.layers import InputLayer
from langflow.schema import Data
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

    def create_sequential (self) -> Data:

        model = Sequential()

        return Data(model=model)
