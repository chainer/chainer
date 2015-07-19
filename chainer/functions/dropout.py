import numpy

from chainer import cuda
from chainer import function


class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio, scale=True):
        self.dropout_ratio = dropout_ratio
        self.scale = scale

    def forward_cpu(self, x):
        if self.scale:
            scale = numpy.float32(1. / (1 - self.dropout_ratio))
        else:
            scale = 1.
        self.mask = scale * \
            (numpy.random.rand(*x[0].shape) >= self.dropout_ratio)
        return x[0] * self.mask,

    def forward_gpu(self, x):
        self.rand = cuda.empty_like(x[0])
        y = cuda.empty_like(x[0])

        cuda.get_generator().fill_uniform(self.rand)
        if self.scale:
            self.scale_val = 1. / (1 - self.dropout_ratio)
        else:
            self.scale_val = 1.

        self.kernel = cuda.elementwise(
            '''float* y, const float* x, const float* rand, float dropout_ratio,
               float scale_val''',
            'y[i] = rand[i] < dropout_ratio ? 0 : scale_val * x[i]',
            'dropout')
        self.kernel(y, x[0], self.rand, self.dropout_ratio, self.scale_val)
        return y,

    def backward_cpu(self, x, gy):
        return gy[0] * self.mask,

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(gy[0])
        self.kernel(gx, gy[0], self.rand, self.dropout_ratio, self.scale_val)
        return gx,


def dropout(x, ratio=.5, train=True, scale=True):
    """Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.
        train (bool): If True, executes dropout. Otherwise, does nothing.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_.

    """
    if train:
        return Dropout(ratio, scale)(x)
    return x
