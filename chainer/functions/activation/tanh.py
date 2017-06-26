import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_TANH


class Tanh(function.Function):

    """Hyperbolic tangent function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        y = utils.force_array(numpy.tanh(x[0]))
        self.retain_inputs(())
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, x):
        if (chainer.should_use_cudnn('==always') and
                x[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            y = cudnn.activation_forward(x[0], _mode)
        else:
            y = cuda.cupy.empty_like(x[0])
            cuda.cupy.tanh(x[0], out=y)
            self.retain_inputs(())

        self.retain_outputs((0,))
        return y,

    def backward_cpu(self, x, gy):
        y = self.output_data[0]
        one = y.dtype.type(1)
        return utils.force_array(gy[0] * (one - y * y)),

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        if (chainer.should_use_cudnn('==always') and
                x[0] is not None and x[0].flags.c_contiguous and
                gy[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            gx = cudnn.activation_backward(x[0], y, gy[0], _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * (1 - y * y)',
                'tanh_bwd')(y, gy[0])
        return gx,


def tanh(x):
    """Elementwise hyperbolic tangent function.

     .. math:: f(x)=\\tanh(x).

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.arange(-1, 4, 2).astype('f')
        >>> x
        array([-1.,  1.,  3.], dtype=float32)
        >>> F.tanh(x).data
        array([-0.76159418,  0.76159418,  0.99505478], dtype=float32)

    """
    return Tanh()(x)
