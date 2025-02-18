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

# TODO: identify the cause of failure of PyTorch when using sigmoid + binary cross entropy

from langflow.custom import Component
from langflow.template import Input, Output
from langflow.schema import Data
import os
import sys

from langflow.io import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    SecretStrInput,
    StrInput,
)

os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"

class KerasSequential (Component):
    display_name = "Keras Sequential"
    description = "Creates a sequential model."
    documentation = "https://keras.io/api/models/sequential/"
    icon = "input"
    name = "KerasSequential"

    inputs = [
        DropdownInput(
            name="input_backend",
            display_name="Backend",
            info="Backend framework function to use.",
            options=["tensorflow", "torch", "jax", "numpy"],
            value="tensorflow",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="create_sequential"),
    ]
    
    # hack because keras cannot dynamically switch backends
    def remove_keras_modules (self):
        keras_modules = [module for module in sys.modules if module.startswith('keras') ]
        for module in keras_modules:
            sys.modules.pop(module, None)
        sys.modules.pop ('tensorflow.python.trackable.data_structures', None)
        print("Keras modules removed:", keras_modules)

    def create_sequential (self) -> Data:
        
        self.remove_keras_modules ()
        os.environ["KERAS_BACKEND"] = self.input_backend

        import keras

        model = keras.Sequential()

        return Data (model=model)
