from onnx import helper

from onnx_chainer import mapping


def convert_AveragePooling2D(
        func, input_names, param_names, parameters, input_tensors):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    gpool = func.inputs[0].shape[2:] == (func.kh, func.kw)
    out_names = [str(id(out())) for out in func.outputs]

    if not gpool:
        return helper.make_node(
            layer_name, input_names, out_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw),
            strides=(func.sy, func.sx)
        ),
    else:
        return helper.make_node(
            'Global' + layer_name, input_names, out_names),
