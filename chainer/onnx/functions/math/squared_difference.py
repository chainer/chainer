from chainer.functions.math.basic_math import PowVarConst
from chainer.functions.math.basic_math import Sub

from onnx_chainer.functions import convert_PowVarConst
from onnx_chainer.functions.math.basic_math import \
    convert_binary_operator  # NOQA


def convert_SquaredDifference(
        func, input_names, param_names, parameters, input_tensors):

    inputs = [i.get_variable() for i in func.inputs]
    sub_func = Sub()
    output = sub_func.apply(inputs)[0]
    sub_node = convert_binary_operator(
        sub_func, input_names, param_names, parameters, input_tensors)[0]

    input_names = [sub_node.output[0]]
    pow_func = PowVarConst(2)
    output = pow_func.apply((output,))[0]
    pow_node = convert_PowVarConst(
        pow_func, input_names, param_names, parameters, input_tensors)[0]

    return list(reversed([sub_node, pow_node]))
