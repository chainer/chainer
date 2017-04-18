import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _algorithm = libcudnn.CUDNN_SOFTMAX_LOG
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL


def logsumexp(x):
    xp = cuda.get_array_module(x)
    m = x.max(axis=1, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=1, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def _log_softmax(x):
    if chainer.should_use_cudnn('>=auto', 3000):
        xp = cuda.get_array_module(x)
        if xp is not numpy:
            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            x_cube = x.reshape(x.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(x_cube)
            y = xp.empty_like(x)
            libcudnn.softmaxForward(
                handle, _algorithm, _mode, one.data, desc.value,
                x_cube.data.ptr, zero.data, desc.value,
                y.data.ptr)
            return y
    log_z = logsumexp(x)
    y = x - log_z
    return y


class LogSoftmax(function.Function):

    """Log-softmax activation function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim > 1,
        )

    def forward(self, xs):
        y = _log_softmax(xs[0])
        self.retain_outputs((0,))
        return y,

    def backward(self, x, gy):
        y = self.output_data[0]
        xp = cuda.get_array_module(*x)
        if xp is not numpy and chainer.should_use_cudnn('>=auto', 3000):
            oz_dtype = 'd' if x[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            gx = xp.empty_like(x[0])
            gx_cube = gx.reshape(gx.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(gx_cube)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, one.data, desc.value,
                y.data.ptr, desc.value, gy[0].data.ptr, zero.data,
                desc.value, gx.data.ptr)
        else:
            gx = gy[0] - xp.exp(y) * gy[0].sum(axis=1, keepdims=True)

        return gx,


def log_softmax(x):
    """Channelwise log-softmax function.

    This function computes its logarithm of softmax along the second axis. Let
    :math:`x = (x_1, x_2, \\dots, x_d)^{\\top}` be the d dimensional index
    array and :math:`f(x_1, \\dots, x_d)` be the corresponding input array.
    For each index :math:`x` in the input array, it computes the logarithm
    of the probability :math:`\log p(x)` defined as

    .. math::
        p(x) = {\\exp(f(x_1, x_2, \\dots, x_d))
                \\over \\sum_{x'_2} \\exp(f(x_1, x'_2, \\dots, x_d))}.

    This method is theoretically equivalent to ``log(softmax(x))`` but is more
    stable.

    .. note::
        ``log(softmax(x))`` may cause underflow when ``x`` is too small,
        because ``softmax(x)`` may returns ``0``.
        ``log_softmax`` method is more stable.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :func:`~chainer.functions.softmax`

    """
    return LogSoftmax()(x)
