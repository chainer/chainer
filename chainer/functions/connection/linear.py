import math

import numpy

from chainer import cuda
from chainer import function
from chainer import link
from chainer.utils import type_check
from chainer import variable


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class LinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        y = x.dot(W.T)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        gy = grad_outputs[0]

        gx = gy.dot(W).reshape(inputs[0].shape)
        gW = gy.T.dot(x)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def linear(x, W, b=None):
    """Linear function, a.k.a. fully-connected layer.

    It accepts two or three arguments: an input minibatch ``x``, an weight
    matrix ``W``, and optionally a bias vector ``b``, and computes
    :math:`Y = xW^\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``..

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`Linear`

    """
    if b is None:
        return LinearFunction()(x, W)
    else:
        return LinearFunction()(x, W, b)


class Linear(link.Link):

    """Linear function with attached parameters.

    This model holds a weight matrix ``W`` and optionally a bias vector ``b``.
    The weight matrix ``W`` has shape ``(out_size, in_size)``. This matrix is
    initialized with i.i.d. Gaussian samples, each of which has zero mean and
    deviation :math:`\sqrt{1/\\text{in_size}}`. The bias vector ``b`` is of
    size ``out_size``. Each element is initialized with the ``bias`` value.
    If ``nobias`` argument is set to True, then this model does not hold a bias
    vector.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    .. seealso:: :func:`~chainer.functions.linear`

    """
    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(Linear, self).__init__()
        if initialW is None:
            initialW = numpy.random.normal(
                0, wscale * math.sqrt(1. / in_size),
                (out_size, in_size)).astype(numpy.float32)
        self.params['W'] = variable.Variable(initialW)

        if not nobias:
            if initial_bias is None:
                initial_bias = numpy.full(out_size, bias, dtype=numpy.float32)
            self.params['b'] = variable.Variable(initial_bias)

    def __call__(self, x):
        return linear(x, self.params['W'], self.params.get('b', None))
