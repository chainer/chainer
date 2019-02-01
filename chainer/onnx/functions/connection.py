import numpy as np
from onnx import helper

import chainer


def convert_Convolution2DFunction(func, onnx_op_name, opset_version,
                                  input_names, output_names, parameters):
    if opset_version == 1:
        if hasattr(func, 'dy') and hasattr(func, 'dx'):
            node = helper.make_node(
                onnx_op_name, input_names, output_names,
                dilations=(func.dy, func.dx),
                kernel_shape=func.inputs[1].shape[2:],
                # pads: [x1_begin, x2_begin...x1_end, x2_end,...]
                pads=(func.ph, func.pw, func.ph, func.pw),
                strides=(func.sy, func.sx),
                group=func.groups,
            )
        else:
            node = helper.make_node(
                onnx_op_name, input_names, output_names,
                dilations=(1, 1),
                kernel_shape=func.inputs[1].shape[2:],
                pads=(func.ph, func.pw, func.ph, func.pw),
                strides=(func.sy, func.sx),
                group=func.groups,
            )
        return node,


def convert_ConvolutionND(func, onnx_op_name, opset_version, input_names,
                          output_names, parameters):
    pad = []
    x_ndim = len(func.inputs[0].shape)
    w_ndim = len(func.inputs[1].shape)
    for _ in range(x_ndim - w_ndim):
        pad.append(0)
    for p in func.pad:
        pad.append(p)
    pad = pad * 2

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.inputs[1].shape[2:],
            pads=pad,
            strides=func.stride,
        ),


def convert_Deconvolution2DFunction(func, onnx_op_name, opset_version,
                                    input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.inputs[1].shape[2:],
            output_shape=(func.outh, func.outw),
            # pads: [x1_begin, x2_begin...x1_end, x2_end,...]
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
        ),


def convert_DeconvolutionND(func, onnx_op_name, opset_version, input_names,
                            output_names, parameters):
    pad = []
    x_ndim = len(func.inputs[0].shape)
    w_ndim = len(func.inputs[1].shape)
    for _ in range(x_ndim - w_ndim):
        pad.append(0)
    for p in func.pad:
        pad.append(p)
    pad = pad * 2

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.inputs[1].shape[2:],
            output_shape=func.outs,
            pads=pad,
            strides=func.stride,
        ),


def convert_EmbedIDFunction(func, onnx_op_name, opset_version, input_names,
                            output_names, parameters):
    x_index_name, W_name = input_names
    input_names = [W_name, x_index_name]

    if func.ignore_label is not None:
        raise ValueError(
            'Current ONNX doesn\'t support ignore_label for EmbedID.')

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, axis=0),


def convert_LinearFunction(func, onnx_op_name, opset_version, input_names,
                           output_names, parameters):
    # When the func has bias
    if len(func.inputs) == 2:
        batchsize = func.inputs[0].shape[0]
        bias_dim = func.inputs[1].shape[0]
        bias = np.zeros((batchsize, bias_dim), dtype=np.float32)
        bias_param = chainer.Parameter(bias)
        parameters.append(bias_param)
        input_names.append(str(id(bias_param)))

    if opset_version == 1 or opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=1.0, beta=1.0, broadcast=1, transA=0, transB=1),
    elif opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            alpha=1.0, beta=1.0, transA=0, transB=1),
