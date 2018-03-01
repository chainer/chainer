import os

import numpy as np

from onnx import helper
from onnx import numpy_helper
from onnx_chainer import mapping


def convert_LinearFunction(
        func, input_names, param_names, parameters, input_tensors):
    input_names[input_names.index(id(func.W))] = param_names[id(func.W)]
    if hasattr(func, 'b'):
        input_names[input_names.index(id(func.b))] = param_names[id(func.b)]
    else:
        # If nobias=True, create zero vector and add it to parameters
        layer_name = os.path.dirname(param_names[id(func.W)])
        bias = np.zeros((func.W.shape[1],), dtype=func.W.array.dtype)
        param_names[id(bias)] = os.path.join(layer_name, 'b')
        parameters.append(
            numpy_helper.from_array(
                bias,
                param_names[id(bias)]
            )
        )
        input_tensors.append(
            helper.make_tensor_value_info(
                param_names[id(bias)],
                mapping.dtypes[bias.dtype],
                bias.shape
            )
        )
        input_names.append(param_names[id(bias)])

    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)
    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        axis=1,
        axis_w=1
    ),
