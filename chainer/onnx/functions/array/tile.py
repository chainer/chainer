import os

import numpy as np

from onnx import helper
from onnx import numpy_helper
from onnx_chainer import mapping


def convert_Tile(func, input_names, param_names, parameters, input_tensors):

    # Add tiles and axis to graph
    if isinstance(func.reps, int):
        func.reps = [func.reps]
    tiles = np.asarray(func.reps, dtype=np.float32)
    axis = np.array([i for i, _ in enumerate(func.reps)], dtype=np.float32)
    layer_name = 'tile_{}'.format(str(id(tiles)))

    param_names[id(tiles)] = os.path.join(layer_name, 'tiles')
    parameters.append(
        numpy_helper.from_array(
            tiles,
            param_names[id(tiles)]
        )
    )
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[id(tiles)],
            mapping.dtypes[tiles.dtype],
            tiles.shape
        )
    )
    input_names.append(param_names[id(tiles)])

    param_names[id(axis)] = os.path.join(layer_name, 'axis')
    parameters.append(
        numpy_helper.from_array(
            axis,
            param_names[id(axis)]
        )
    )
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[id(axis)],
            mapping.dtypes[axis.dtype],
            axis.shape
        )
    )
    input_names.append(param_names[id(axis)])

    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    axis = [i for i, _ in enumerate(func.reps)]

    return helper.make_node(layer_name, input_names, out_names),
