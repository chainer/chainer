import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _algorithm = libcudnn.CUDNN_SOFTMAX_ACCURATE
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL


class Softmax(function.Function):

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
        if (xp is not numpy and chainer.should_use_cudnn('>=auto') and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            oz_dtype = 'd' if x[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            x_tensor4d = x[0].reshape(self._get_tensor4d_shape(x[0].shape))
            desc = cudnn.create_tensor_descriptor(x_tensor4d)
            y = xp.empty_like(x[0])
            libcudnn.softmaxForward(
                handle, _algorithm, _mode, one.data, desc.value,
                x_tensor4d.data.ptr, zero.data, desc.value,
                y.data.ptr)
        else:
            y = x[0] - x[0].max(axis=self.axis, keepdims=True)
            xp.exp(y, out=y)
            y /= y.sum(axis=self.axis, keepdims=True)

        self._x_shape = x[0].shape
        self.retain_inputs(())
        self.retain_outputs((0,))
        return y,

    def backward(self, x, gy):
        y = self.output_data[0]
        xp = cuda.get_array_module(*y)
        if (xp is not numpy and chainer.should_use_cudnn('>=auto') and
                (_cudnn_version >= 3000 or y.dtype != numpy.float16)):
            oz_dtype = 'd' if y[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            gx = xp.empty_like(y)
            gx_tensor4d = gx.reshape(self._get_tensor4d_shape(gx.shape))
            desc = cudnn.create_tensor_descriptor(gx_tensor4d)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, one.data, desc.value,
                y.data.ptr, desc.value, gy[0].data.ptr, zero.data,
                desc.value, gx.data.ptr)
        else:
            gx = y * gy[0]
            sumdx = gx.sum(axis=self.axis, keepdims=True)
            gx -= y * sumdx

        return gx,

    def _get_tensor4d_shape(self, shape):
        left_shape = numpy.prod(shape[slice(0, self.axis)], dtype=numpy.int)
        center_shape = shape[self.axis]
        right_shape = numpy.prod(
            shape[slice(self.axis + 1, len(shape))], dtype=numpy.int)
        return left_shape, center_shape, right_shape, 1


def softmax(x, axis=1):
    """Softmax function.

    This function computes its softmax along an axis. Let
    :math:`x = (x_1, x_2, \\dots, x_D)^{\\top}` be the D dimensional index
    array and :math:`f(x)` be the D dimensional input array. For each index
    :math:`x` of the input array :math:`f(x)`, it computes the probability
    :math:`p(x)` defined as
    :math:`p(x) = {\\exp(f(x)) \\over \\sum_{d} \\exp(f(x_d))}`.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Softmax(axis=axis)(x)
