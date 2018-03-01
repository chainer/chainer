from onnx import helper

from onnx_chainer import mapping


def convert_Pad(func, input_names, param_names, parameters, input_tensors):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    if func.mode not in ['constant', 'reflect', 'edge']:
        raise ValueError(
            '{} mode is not supported in ONNX\'s Pad operation'.format(
                func.mode))

    if 'constant_values' in func.keywords:
        values = func.keywords['constant_values']
        if not isinstance(values, int) and len(values) > 1:
            raise ValueError(
                'ONNX doesn\'t support multiple constant values for Pad '
                'operation')
        elif not isinstance(values, int):
            values = values[0]

        node = helper.make_node(
            layer_name, input_names, out_names,
            mode=func.mode,
            pads=func.pad_bw.tolist(),
            value=values
        )
    else:
        node = helper.make_node(
            layer_name, input_names, out_names,
            mode=func.mode,
            pads=func.pad_bw.tolist(),
        )

    return node,
