import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_RELU


class ReLU(function.Function):

    """Rectified Linear Unit."""

    def __init__(self, use_cudnn=True, inplace=False):
        self.use_cudnn = use_cudnn
        self.inplace = inplace

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        if self.inplace:
            out = x[0]
        else:
            out = numpy.empty_like(x[0])
        return numpy.maximum(x[0], 0, out=out),

    def forward_gpu(self, x):
        if self.inplace:
            y = x[0]
        else:
            y = cuda.cupy.empty_like(x[0])

        if (cuda.cudnn_enabled and self.use_cudnn and
                x[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            self.y = cuda.cudnn.activation_forward(x[0], _mode, out=y)
        else:
            cuda.cupy.maximum(x[0], 0, out=y)

        return y,

    def backward_cpu(self, x, gy):
        if self.inplace:
            gx = gy[0]
        else:
            gx = numpy.empty_like(gy[0])
        gx[...] = gy[0] * (x[0] > 0)
        return gx,

    def backward_gpu(self, x, gy):
        if self.inplace:
            gx = gy[0]
        else:
            gx = cuda.cupy.empty_like(gy[0])

        if (cuda.cudnn_enabled and self.use_cudnn and
                x[0].flags.c_contiguous and gy[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            cudnn.activation_backward(x[0], self.y, gy[0], _mode, out=gx)
        else:
            kernel = cuda.elementwise(
                'T x, T gy', 'T gx',
                'gx = x > 0 ? gy : (T)0',
                'relu_bwd')
            kernel(x[0], gy[0], gx)

        return gx,


def relu(x, use_cudnn=True, inplace=False):
    """Rectified Linear Unit function.

    .. math:: f(x)=\\max(0, x).

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.
        inplace (bool): If ``True``, this function does in-place calculation.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> np.any(x < 0)
        True
        >>> y = F.relu(x)
        >>> np.any(y.data < 0)
        False
        >>> y.shape
        (3, 2)

    """
    return ReLU(use_cudnn, inplace)(x)
