import chainer
import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((9,))
def convert_SoftmaxCrossEntropy(
        func, opset_version, input_names, output_names, context):
    # obtain input variable
    if not isinstance(func, chainer.FunctionNode):
        raise NotImplementedError(
            'SoftmaxCrossEntropy is currently supported for Chainer>=6.0.0a1.')

    x_var, t_var = func.get_retained_inputs()
    if len(x_var.shape) != 2:
        raise NotImplementedError(
            'ONNX-Chainer currently handles SoftmaxCrossEntropy only when '
            'the dimension of input variable x is exactly two.')
    if np.any(t_var.array == func.ignore_label):
        raise NotImplementedError(
            'ONNX-Chainer currently handles SoftmaxCrossEntropy only when '
            'ignore_label is not used in input variable t.')
    if (not func.normalize) or (func.class_weight is not None) or\
       (func.ignore_label != -1) or (func.reduce != 'mean'):
        raise NotImplementedError(
            'ONNX-Chainer currently handles SoftmaxCrossEntropy only when '
            'argument parameters are default setting.')

    # create intermediate values
    gb = onnx_helper.GraphBuilder()
    x, t = input_names
    y_log = gb.op('LogSoftmax', [x])
    depth = context.add_const(np.array([x_var.shape[1]], dtype=np.int32),
                              'depth')
    zeroone = context.add_const(np.array([0, 1], dtype=x_var.dtype), 'zeroone')
    th = gb.op('OneHot', [t, depth, zeroone])
    s0 = gb.op('Mul', [y_log, th])
    sn = gb.op('Neg', [s0])
    sr = gb.op('ReduceSum', [sn], axes=[1], keepdims=0)
    gb.op_output_named('ReduceMean', [sr], output_names, axes=[0], keepdims=0)

    return gb.nodes()
