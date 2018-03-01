import numpy as np

from onnx import helper
from onnx import numpy_helper
from onnx_chainer import mapping

import os


def convert_unary_operator(
        func, input_names, param_names, parameters, input_tensors):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(layer_name, input_names, out_names),


def convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(layer_name, input_names, out_names),


def convert_Add(func, input_names, param_names, parameters, input_tensors):
    return convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors)


def convert_Sub(func, input_names, param_names, parameters, input_tensors):
    return convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors)


def convert_Mul(func, input_names, param_names, parameters, input_tensors):
    return convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors)


def convert_Neg(func, input_names, param_names, parameters, input_tensors):
    return convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors)


def convert_Div(func, input_names, param_names, parameters, input_tensors):
    return convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors)


def convert_Absolute(
        func, input_names, param_names, parameters, input_tensors):
    return convert_unary_operator(
        func, input_names, param_names, parameters, input_tensors)


def convert_PowVarConst(
        func, input_names, param_names, parameters, input_tensors):
    layer_name = 'Pow_{}'.format(str(id(func.value)))
    value = np.asarray([func.value], dtype=func.inputs[0].get_variable().dtype)
    param_names[id(func.value)] = os.path.join(layer_name, 'value')

    parameters.append(
        numpy_helper.from_array(
            value,
            param_names[id(func.value)]
        )
    )
    input_tensors.append(
        helper.make_tensor_value_info(
            param_names[id(func.value)],
            mapping.dtypes[value.dtype],
            value.shape
        )
    )
    input_names.append(param_names[id(func.value)])

    return convert_binary_operator(
        func, input_names, param_names, parameters, input_tensors)
