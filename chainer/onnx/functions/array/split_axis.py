from onnx import helper

from onnx_chainer import mapping


def convert_SplitAxis(
        func, input_names, param_names, parameters, input_tensors):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = mapping.operators[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    if hasattr(func.indices_or_sections, '__iter__'):
        split = []
        prev_i = 0
        for i in func.indices_or_sections:
            split.append(i - prev_i)
            prev_i = i
    else:
        length = func.inputs[0].shape[func.axis] // func.indices_or_sections
        split = [length for _ in range(func.indices_or_sections)]

    return helper.make_node(
        layer_name, input_names, out_names,
        axis=func.axis,
        split=split
    ),
