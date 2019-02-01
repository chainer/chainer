from onnx import helper


def convert_AveragePooling2D(func, onnx_op_name, opset_version, input_names,
                             output_names, parameters):
    pad = [func.ph, func.pw]
    stride = [func.sy, func.sx]
    ksize = [func.kh, func.kw]
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, stride, ksize):
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        raise ValueError(
            'AveragePooling2D is not compatible with ONNX\'s AveragePool-1. '
            'Use operation set version >= 7.')
    elif opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=ksize,
            pads=pad,
            strides=stride,
            count_include_pad=1,
        ),


def convert_AveragePoolingND(func, onnx_op_name, opset_version, input_names,
                             output_names, parameters):
    pad = list(func.pad[:])
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, func.stride, func.ksize):
            # Raise exception because a virtual pad for cover_all must be
            # smaller than ksize in the current ONNX
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, func.stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        raise ValueError(
            'AveragePoolingND is not compatible with ONNX\'s AveragePool-1. '
            'Use operation set version >= 7.')
    elif opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride,
            count_include_pad=1,
        ),


def convert_MaxPooling2D(func, onnx_op_name, opset_version, input_names,
                         output_names, parameters):
    pad = [func.ph, func.pw]
    stride = [func.sy, func.sx]
    ksize = [func.kh, func.kw]
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, stride, ksize):
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=ksize,
            pads=pad,
            strides=stride
        ),
    elif opset_version == 8:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=ksize,
            pads=pad,
            strides=stride,
            storage_order=0,  # row major
        ),


def convert_MaxPoolingND(func, onnx_op_name, opset_version, input_names,
                         output_names, parameters):
    pad = list(func.pad[:])
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, func.stride, func.ksize):
            # Raise exception because a virtual pad for cover_all must be
            # smaller than ksize in the current ONNX
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, func.stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride
        ),
    elif opset_version == 8:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride,
            storage_order=0,  # row major
        ),
