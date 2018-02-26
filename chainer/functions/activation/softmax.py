import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _algorithm = libcudnn.CUDNN_SOFTMAX_ACCURATE
    _mode_channel = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL
    _mode_instance = libcudnn.CUDNN_SOFTMAX_MODE_INSTANCE


def _get_tensor4d_shape(axis, shape):
    left_shape = numpy.prod(shape[slice(0, axis)], dtype=numpy.int)
    center_shape = shape[axis]
    right_shape = numpy.prod(
        shape[slice(axis + 1, len(shape))], dtype=numpy.int)
    return left_shape, center_shape, right_shape, 1


def _get_cudnn_mode(shape):
    if shape[2] == 1 and shape[3] == 1:
        return _mode_instance
    return _mode_channel


class Softmax(function_node.FunctionNode):

    """Softmax activation function."""

    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim > 1,
            self.axis < x_type.ndim
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        if xp is not numpy and chainer.should_use_cudnn('>=auto'):
            oz_dtype = 'd' if x[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            x_tensor4d = cuda.cupy.ascontiguousarray(
                x[0].reshape(_get_tensor4d_shape(self.axis, x[0].shape)))
            desc = cudnn.create_tensor_descriptor(x_tensor4d)
            cudnn_mode = _get_cudnn_mode(x_tensor4d.shape)
            y = xp.empty_like(x[0])
            libcudnn.softmaxForward(
                handle, _algorithm, cudnn_mode, one.data, desc.value,
                x_tensor4d.data.ptr, zero.data, desc.value,
                y.data.ptr)
        else:
            y = x[0] - x[0].max(axis=self.axis, keepdims=True)
            xp.exp(y, out=y)
            y /= y.sum(axis=self.axis, keepdims=True)

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        y = self.get_retained_outputs()[0]
        gy, = grad_outputs
        return _SoftmaxGrad(self.axis).apply((y, gy))


class _SoftmaxGrad(function_node.FunctionNode):

    def __init__(self, axis):
        self.axis = axis

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        xp = cuda.get_array_module(*y)
        if xp is not numpy and chainer.should_use_cudnn('>=auto'):
            oz_dtype = 'd' if y[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            gx = xp.empty_like(y)
            gx_tensor4d = cuda.cupy.ascontiguousarray(
                gx.reshape(_get_tensor4d_shape(self.axis, gx.shape)))
            gy = cuda.cupy.ascontiguousarray(gy)
            desc = cudnn.create_tensor_descriptor(gx_tensor4d)
            cudnn_mode = _get_cudnn_mode(gx_tensor4d.shape)
            libcudnn.softmaxBackward(
                handle, _algorithm, cudnn_mode, one.data, desc.value,
                y.data.ptr, desc.value, gy.data.ptr, zero.data,
                desc.value, gx.data.ptr)
        else:
            gx = y * gy
            sumdx = gx.sum(axis=self.axis, keepdims=True)
            gx -= y * sumdx

        return gx,

    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        ggx, = grad_outputs
        gs = chainer.functions.sum(ggx * y, axis=self.axis, keepdims=True)
        ga = ggx - chainer.functions.broadcast_to(gs, gy.shape)
        ret = []
        if 0 in indexes:
            s = chainer.functions.broadcast_to(chainer.functions.sum(
                y * gy, axis=self.axis, keepdims=True), gy.shape)
            gy2 = ga * gy - ggx * s
            ret.append(gy2)
        if 1 in indexes:
            ggy = ga * y
            ret.append(ggy)
        return tuple(ret)


def softmax(x, axis=1):
    """Softmax function.

    This function computes its softmax along an axis. Let
    :math:`c = (c_1, c_2, \\dots, c_D)` be the slice of ``x`` along with
    the axis. For each slice :math:`c`, it computes the function :math:`f(c)`
    defined as :math:`f(c)={\\exp(c) \\over \\sum_{d} \\exp(c_d)}`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable.
            A :math:`n`-dimensional (:math:`n \\geq 2`) float array.
        axis (int): The axis along which the softmax is to be computed.

    Returns:
        ~chainer.Variable: Output variable.
        A :math:`n`-dimensional (:math:`n \\geq 2`) float array, which is the
        same shape with x.

    .. admonition:: Example

        >>> x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        >>> x
        array([[0., 1., 2.],
               [0., 2., 4.]], dtype=float32)
        >>> y = F.softmax(x, axis=1)
        >>> y.data
        array([[0.09003057, 0.24472848, 0.66524094],
               [0.01587624, 0.11731043, 0.86681336]], dtype=float32)
        >>> F.sum(y, axis=1).data
        array([1., 1.], dtype=float32)

    """
    return Softmax(axis=axis).apply((x,))[0]
