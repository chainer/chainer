import numpy
import six

from chainer import cuda
from chainer import function
from chainer import link
from chainer.utils import type_check
from chainer import variable


def _fwd_kern():
    return cuda.elementwise(
        'T x, T cond, T W', 'T y',
        'y = cond >= 0 ? x : x * W', 'prelu')


class PReLUFunction(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, W_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            W_type.dtype == numpy.float32,
            x_type.ndim >= W_type.ndim + 1,
            x_type.shape[1: 1 + W_type.ndim.eval()] == W_type.shape
        )

    def forward_cpu(self, inputs):
        x, W = inputs
        y = x.copy()
        masked = numpy.ma.masked_greater_equal(y, 0, copy=False)
        shape = _get_extended_shape(W, y)
        masked *= W.reshape(shape)
        return y,

    def forward_gpu(self, inputs):
        x, W = inputs
        shape = _get_extended_shape(W, x)
        y = _fwd_kern()(x, x, W.reshape(shape))
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs
        gy = grad_outputs[0]
        mask = x >= 0
        masked_x_gy = numpy.ma.array(x * gy, mask=mask)
        axes = (0,) + tuple(six.moves.range(1 + W.ndim, gy.ndim))
        gW = masked_x_gy.sum(axis=axes)
        if numpy.isscalar(gW):
            gW = numpy.array(gW)

        gx = gy.copy()
        masked = numpy.ma.array(gx, mask=mask)
        shape = _get_extended_shape(W, gx)
        masked *= W.reshape(shape)

        return gx, gW

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs
        gy = grad_outputs[0]
        masked = cuda.elementwise(
            'T x, T gy', 'T masked',
            'masked = x >= 0 ? 0 : x * gy',
            'prelu_masked')(x, gy)
        axes = (0,) + tuple(six.moves.range(1 + W.ndim, gy.ndim))
        gW = masked.sum(axis=axes)

        gx = masked  # reuse buffer
        shape = _get_extended_shape(W, gx)
        _fwd_kern()(gy, x, W.reshape(shape), gx)
        return gx, gW


def prelu(x, W):
    """Parametric ReLU function.

    It accepts two arguments: an input ``x`` and a weight array ``W``.
    PReLU function is written in elementwise equation as
    :math:`PReLU(x) = \max(x, ax)`, where :math:`a` is a parameter array.

    When the PReLU function is combined with two-dimensional convolution, the
    elements of parameter :math:`a` are typically shared across the same filter
    of different pixels. In order to support such usage, this function supports
    the shape of parameter array that indicates leading dimensions of input
    arrays except the batch dimension.

    Args:
        x (~chainer.Variable): Input variable.
    Its first argument is assumed to be the minibatch dimension.
        W (~chainer.Variable): Weight variable.

    Returns:
        ~chainer.Variable: Output variable

    """
    return PReLUFunction()(x, W)


class PReLU(link.Link):
    """Parametric ReLU function with attached parameters.

    Args:
        shape (tuple of ints): Shape of the parameter array.
        init (float): Initial parameter value.

    See detail in paper: `Delving Deep into Rectifiers: Surpassing \
    Human-Level Performance on ImageNet Classification \
    <http://arxiv.org/abs/1502.01852>`_.

    """

    def __init__(self, shape=(), init=0.25):
        super(PReLU, self).__init__()
        self.params['W'] = variable.Variable(
            numpy.full(shape, init, dtype=numpy.float32))

    def __call__(self, x):
        return prelu(x, self.params['W'])


def _get_extended_shape(W, x):
    return (1,) + W.shape + (1,) * (x.ndim - W.ndim - 1)
