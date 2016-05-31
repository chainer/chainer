import numpy

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

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.y = utils.force_array(numpy.tanh(x[0]))
        return self.y,

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            self.y = cudnn.activation_forward(x[0], _mode)
        else:
            self.y = cuda.cupy.empty_like(x[0])
            cuda.cupy.tanh(x[0], out=self.y)
        return self.y,

    def backward_cpu(self, x, gy):
        one = x[0].dtype.type(1)
        return utils.force_array(gy[0] * (one - self.y * self.y)),

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            gx = cudnn.activation_backward(x[0], self.y, gy[0], _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * (1 - y * y)',
                'tanh_bwd')(self.y, gy[0])
        return gx,


def tanh(x, use_cudnn=True):
    """Elementwise hyperbolic tangent function.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Tanh(use_cudnn)(x)
