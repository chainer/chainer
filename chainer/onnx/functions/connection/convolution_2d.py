from onnx import helper

from onnx_chainer import mapping


def convert_Convolution2DFunction(
        func, input_names, param_names, parameters, input_tensors):
    input_names[input_names.index(id(func.W))] = param_names[id(func.W)]
    if hasattr(func, 'b'):
        input_names[input_names.index(id(func.b))] = param_names[id(func.b)]
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    if hasattr(func, 'dy') and hasattr(func, 'dx'):
        node = helper.make_node(
            layer_name, input_names, out_names,
            dilations=(func.dy, func.dx),
            kernel_shape=func.W.shape[2:],
            # pads: [x1_begin, x2_begin...x1_end, x2_end,...]
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
        )
    else:
        node = helper.make_node(
            layer_name, input_names, out_names,
            dilations=(1, 1),
            kernel_shape=func.W.shape[2:],
            pads=(func.ph, func.ph, func.pw, func.pw),
            strides=(func.sy, func.sx),
        )

    return node,
