from onnx import helper

from onnx_chainer import mapping


def convert_EmbedIDFunction(
        func, input_names, param_names, parameters, input_tensors):
    input_names[input_names.index(id(func.W))] = param_names[id(func.W)]
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)
    del input_names[1]

    for i, param in enumerate(parameters):
        if param.name == param_names[id(func.W)]:
            break
    del param_names[id(func.W)]
    del parameters[i]

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    if func.ignore_label is not None:
        raise ValueError(
            'Current ONNX doesn\'t support ignore_label for EmbedID.')

    return helper.make_node(
        layer_name, input_names, out_names,
        input_dim=func._w_shape[0],
        output_dim=func._w_shape[1],
        weights=param,
    ),
