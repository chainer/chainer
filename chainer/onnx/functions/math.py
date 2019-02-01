import numpy as np
from onnx import helper

import chainer


def convert_Add(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_AddConstant(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    value = np.asarray([func.value], dtype=func.inputs[0].dtype)
    value = np.broadcast_to(value, func.inputs[0].shape)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(str(id(value_param)))

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sub(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Mul(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_MulConstant(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    value = np.asarray([func.value], dtype=func.inputs[0].dtype)
    value = np.broadcast_to(value, func.inputs[0].shape)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(str(id(value_param)))

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Neg(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Div(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Absolute(func, onnx_op_name, opset_version, input_names,
                     output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_PowVarConst(func, onnx_op_name, opset_version, input_names,
                        output_names, parameters):
    value = np.asarray([func.value], dtype=func.inputs[0].dtype)
    value = np.broadcast_to(value, func.inputs[0].shape)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(str(id(value_param)))

    if opset_version == 1 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Clip(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            max=func.x_max,
            min=func.x_min,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            max=func.x_max,
            min=func.x_min,
        ),


def convert_Exp(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Identity(func, onnx_op_name, opset_version, input_names,
                     output_names, parameters):
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_MatMul(func, onnx_op_name, opset_version, input_names,
                   output_names, parameters):
    bias_shape = (
        func.inputs[0].shape[-1] if func.transa else func.inputs[0].shape[-2],
        func.inputs[1].shape[-2] if func.transb else func.inputs[1].shape[-1]
    )
    bias_tensor = np.zeros(bias_shape, dtype=np.float32)
    bias_param = chainer.Parameter(bias_tensor)
    parameters.append(bias_param)
    input_names.append(str(id(bias_param)))

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        transA=func.transa,
        transB=func.transb
    ),


def convert_Maximum(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 8:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Minimum(func, onnx_op_name, opset_version, input_names,
                    output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 8:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sqrt(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_LogSumExp(func, onnx_op_name, opset_version, input_names,
                      output_names, parameters):
    # Use keepdims=False by default
    # since the chainer does not support keepdims option
    if hasattr(func, 'keepdims'):
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axes=func.axis,
            keepdims=func.keepdims,
        ),
    else:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            axes=func.axis,
            keepdims=False,
        ),


def convert_Max(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),


def convert_Mean(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),


def convert_Min(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),


def convert_Prod(func, onnx_op_name, opset_version, input_names, output_names,
                 parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),


def convert_Sum(func, onnx_op_name, opset_version, input_names, output_names,
                parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),


def convert_LinearInterpolate(func, onnx_op_name, opset_version, input_names,
                              output_names, parameters):
    typ = func.inputs[0].dtype if isinstance(
        func.inputs[0].dtype, np.dtype) else np.dtype(func.inputs[0].dtype)

    one = chainer.Parameter(np.array(1, dtype=typ))
    parameters.append(one)

    kwargs = {"consumed_inputs": [1, 1]} if opset_version == 1 else {}
    kwargs2 = {} if opset_version >= 7 else {"broadcast": 1}

    n1_out_name = gensym()
    n2_out_name = gensym()
    n3_out_name = gensym()

    n1 = helper.make_node(
        "Sub", [str(id(one)), input_names[0]], [n1_out_name],
        **kwargs, **kwargs2)
    n2 = helper.make_node(
        "Mul", [input_names[0], input_names[1]], [n2_out_name], **kwargs)
    n3 = helper.make_node(
        "Mul", [n1_out_name, input_names[2]], [n3_out_name], **kwargs)
    n4 = helper.make_node(
        "Add", [n2_out_name, n3_out_name], [output_names[0]], **kwargs)

    return n4, n3, n2, n1


dummy_objects = []


def gensym():
    o = object()
    dummy_objects.append(o)
    return str(id(o))
