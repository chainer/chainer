import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


def _convert_softmax_impl(op_type, func, input_names, output_names):
    axis = func.axis
    ndim = len(func.inputs[0].shape)
    if axis == ndim - 1:
        return onnx_helper.make_node(
            op_type, input_names, output_names,
            axis=axis
        ),

    # Chainer's softmax computes the softmax along a single axis while
    # ONNX's computes along the specified axis and all axes after the
    # specified axis. To emulate Chainer's by ONNX's, we transpose the
    # single specified axis to the last axis, compute the softmax, and
    # transpose back to the original shape.
    gb = onnx_helper.GraphBuilder()
    perm = list(range(ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    transposed = gb.op('Transpose', input_names, perm=perm)
    softmaxed = gb.op(op_type, [transposed], axis=ndim - 1)
    gb.op('Transpose', [softmaxed], perm=perm)
    return gb.nodes(output_names=output_names)


@support((1, 6, 11))
def convert_ClippedReLU(
        func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Clip', input_names, output_names,
            min=0.0, max=func.cap,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Clip', input_names, output_names,
            min=0.0, max=func.cap,
        ),
    elif opset_version == 11:
        min_name = context.add_const(np.array(0, dtype=np.float32), 'zero')
        max_name = context.add_const(
            np.array(func.cap, dtype=np.float32), 'clip_z')
        input_names.extend([min_name, max_name])
        return onnx_helper.make_node('Clip', input_names, output_names),


@support((1, 6))
def convert_ELU(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Elu', input_names, output_names,
            alpha=func.alpha,
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Elu', input_names, output_names,
            alpha=func.alpha
        ),


@support((1, 6))
def convert_HardSigmoid(
        func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'HardSigmoid', input_names, output_names,
            alpha=0.2,
            beta=0.5,
            consumed_inputs=[1],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'HardSigmoid', input_names, output_names,
            alpha=0.2,
            beta=0.5
        ),


@support((1, 6))
def convert_LeakyReLU(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'LeakyRelu', input_names, output_names,
            alpha=func.slope,
            consumed_inputs=[1],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'LeakyRelu', input_names, output_names,
            alpha=func.slope
        ),


def convert_LogSoftmax(
        func, opset_version, input_names, output_names, context):
    return _convert_softmax_impl('LogSoftmax', func, input_names, output_names)


@support((1, 6, 7))
def convert_PReLUFunction(
        func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'PRelu', input_names, output_names, consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('PRelu', input_names, output_names),
    elif opset_version == 7:
        return onnx_helper.make_node('PRelu', input_names, output_names),


@support((1, 6))
def convert_ReLU(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Relu', input_names, output_names,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Relu', input_names, output_names),


@support((1, 6))
def convert_Selu(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Selu', input_names, output_names,
            consumed_inputs=[1],
            alpha=func.alpha,
            gamma=func.scale
        ),
    elif opset_version == 6:
        return onnx_helper.make_node('Selu', input_names, output_names,
                                     alpha=func.alpha,
                                     gamma=func.scale
                                     ),


@support((1, 6))
def convert_Sigmoid(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Sigmoid', input_names, output_names,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Sigmoid', input_names, output_names),


def convert_Softmax(func, opset_version, input_names, output_names, context):
    return _convert_softmax_impl('Softmax', func, input_names, output_names)


def convert_Softplus(func, opset_version, input_names, output_names, context):
    return onnx_helper.make_node('Softplus', input_names, output_names),


@support((1, 6))
def convert_Tanh(func, opset_version, input_names, output_names, context):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Tanh', input_names, output_names,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Tanh', input_names, output_names),
