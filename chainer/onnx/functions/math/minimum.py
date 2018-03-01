from onnx_chainer.functions.math.basic_math import convert_unary_operator  # NOQA


def convert_Minimum(
        func, input_names, param_names, parameters, input_tensors):
    return convert_unary_operator(
        func, input_names, param_names, parameters, input_tensors)
