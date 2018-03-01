from onnx import helper

from onnx_chainer import mapping


def convert_Deconvolution2DFunction(
        func, input_names, param_names, parameters, input_tensors):
    input_names[input_names.index(id(func.W))] = param_names[id(func.W)]
    if hasattr(func, 'b'):
        input_names[input_names.index(id(func.b))] = param_names[id(func.b)]
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        auto_pad='VALID',
        kernel_shape=func.W.shape[2:],
        output_shape=(func.outh, func.outw),
        pads=(func.ph, func.pw),
        strides=(func.sy, func.sx),
    ),
