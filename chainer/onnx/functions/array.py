import numpy as np
from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

import chainer
from onnx_chainer import mapping


def convert_Cast(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    typ = func.type if isinstance(func.type, np.dtype) else np.dtype(func.type)
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            to=mapping.TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            to=NP_TYPE_TO_TENSOR_TYPE[typ]
        ),


def convert_Concat(func, onnx_op_name, opset_version, input_names,
                   output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axis=func.axis
        ),
    elif opset_version == 4:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axis=func.axis
        ),


def convert_Copy(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names
    ),


def convert_Depth2Space(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_GetItem(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        slice=func.slices
    ),


def convert_Pad(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if func.mode not in ['constant', 'reflect', 'edge']:
        raise ValueError(
            '{} mode is not supported in ONNX\'s Pad operation'.format(
                func.mode))

    pad_begin = []
    pad_end = []
    for pp in func.pad_bw.tolist():
        pad_begin.append(pp[0])
        pad_end.append(pp[1])
    pad = pad_begin + pad_end

    if 'constant_values' in func.keywords:
        values = func.keywords['constant_values']
        if not isinstance(values, int) and len(values) > 1:
            raise ValueError(
                'ONNX doesn\'t support multiple constant values for Pad '
                'operation')
        elif not isinstance(values, int):
            values = values[0]

        if opset_version == 1:
            node = helper.make_node(
                onnx_op_name, input_names, output_names,
                mode=func.mode,
                paddings=pad,
                value=values
            )
        elif opset_version == 2:
            node = helper.make_node(
                onnx_op_name, input_names, output_names,
                mode=func.mode,
                pads=pad,
                value=values
            )
    else:
        if opset_version == 1:
            node = helper.make_node(
                onnx_op_name, input_names, output_names,
                mode=func.mode,
                paddings=pad,
                value=0.,
            )
        elif opset_version == 2:
            node = helper.make_node(
                onnx_op_name, input_names, output_names,
                mode=func.mode,
                pads=pad,
                value=0.,
            )

    return node,


def convert_Reshape(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            shape=func.shape
        ),
    elif opset_version == 5:
        shape = np.asarray(list(func.shape), dtype=np.int64)
        shape_param = chainer.Parameter(shape)
        parameters.append(shape_param)
        input_names.append(str(id(shape_param)))

        return helper.make_node(
            onnx_op_name, input_names, output_names,
        ),


def convert_Space2Depth(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_SplitAxis(func, onnx_op_name, opset_version, input_names,
                      output_names, parameters):
    if func.indices is not None:
        indices_or_sections = func.indices
    else:
        indices_or_sections = func.sections

    if hasattr(indices_or_sections, '__iter__'):
        split = []
        prev_i = 0
        for i in indices_or_sections:
            split.append(i - prev_i)
            prev_i = i
    else:
        length = func.inputs[0].shape[func.axis] // indices_or_sections
        split = [length for _ in range(indices_or_sections)]

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axis=func.axis,
            split=split
        ),
    elif opset_version == 2:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axis=func.axis,
            split=split
        ),


def convert_Squeeze(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    if func.axis is None:
        axis = []
        for i, s in enumerate(func.inputs[0].shape):
            if s == 1:
                axis.append(i)
    else:
        axis = func.axis

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=axis
    ),


def convert_Tile(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    # Add tiles and axis to graph
    if isinstance(func.reps, int):
        func.reps = [func.reps]
    tiles = np.asarray(func.reps, dtype=np.int64)

    tiles_param = chainer.Parameter(tiles)
    parameters.append(tiles_param)
    input_names.append(str(id(tiles_param)))

    # In operater version = 1, axis also should be given
    if opset_version == 1:
        axis = np.array([i for i, _ in enumerate(func.reps)], dtype=np.float32)
        axis_param = chainer.Parameter(axis)
        parameters.append(axis_param)
        input_names.append(str(id(axis_param)))

    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Transpose(func, onnx_op_name, opset_version, input_names,
                      output_names, parameters):

    if func.axes is None:
        node = helper.make_node(onnx_op_name, input_names, output_names)
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            perm=func.axes
        )

    return node,
