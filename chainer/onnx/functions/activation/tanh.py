from onnx import helper

from onnx_chainer import mapping


def convert_Tanh(
        func, input_names, param_names, parameters, input_tensors):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]
    return helper.make_node(layer_name, input_names, out_names),
