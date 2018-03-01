from onnx_chainer.functions.math.basic_math import convert_unary_operator  # NOQA


def convert_MatMul(
        func, input_names, param_names, parameters, input_tensors):
    if func.transa or func.transb:
        raise ValueError(
            'Current ONNX doesn\'t support transpose options for matmul ops.')
    return convert_unary_operator(
        func, input_names, param_names, parameters, input_tensors)
