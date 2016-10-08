import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Sin(function.Function):

    @property
    def label(self):
        return 'sin'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sin(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.cos(x[0]))
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = cos(x) * gy', 'sin_bwd'
        )(x[0], gy[0])
        return gx,


def sin(x):
    """Elementwise sin function.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sin()(x)


class Cos(function.Function):

    @property
    def label(self):
        return 'cos'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.cos(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.sin(x[0]))
        numpy.negative(gx, out=gx)
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = -sin(x) * gy', 'cos_bwd'
        )(x[0], gy[0])
        return gx,


def cos(x):
    """Elementwise cos function.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cos()(x)


class Tan(function.Function):

    @property
    def label(self):
        return 'tan'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.tan(x[0])),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.cos(x[0]))
        xp.square(gx, out=gx)
        xp.reciprocal(gx, out=gx)
        gx *= gy[0]
        return gx,


def tan(x):
    """Elementwise tan function.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Tan()(x)
