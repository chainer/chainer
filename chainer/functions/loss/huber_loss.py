import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_mat(x, n_batch_axes):
    if x.ndim == 2:
        return x
    elif n_batch_axes < 1:
        raise ValueError('n_batch_axes should be greater than 0 but {} '
                         'was given.'.format(n_batch_axes))
    elif n_batch_axes == 1:
        return x.reshape(len(x), -1)
    elif n_batch_axes < x.ndim:
        x.reshape(x.shape[:n_batch_axes] + (-1,))
    else:
        raise ValueError('n_batch_axes should be less than x.ndim but {} '
                         'was given.'.format(n_batch_axes))


class HuberLoss(function.Function):

    def __init__(self, delta, n_batch_axes=1):
        self.delta = delta
        self.n_batch_axes = n_batch_axes

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1 = inputs
        self.diff = x0 - x1
        y = xp.square(self.diff)
        mask = y > (self.delta ** 2)
        y -= mask * xp.square(abs(self.diff) - self.delta)
        y *= 0.5
        return y.sum(axis=1),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        mask = xp.abs(self.diff) <= self.delta
        gx = gy[0].reshape(gy[0].shape + (1,) * (self.diff.ndim - 1)) * \
            xp.where(mask, self.diff, self.delta * xp.sign(self.diff))
        return gx, -gx


def huber_loss(x, t, delta, n_batch_axes=1):
    """Loss function which is less sensitive to outliers in data than MSE.

        .. math::
            a = x - t

        and

        .. math::
            L_{\\delta}(a) = \\left \\{ \\begin{array}{cc}
            \\frac{1}{2} a^2 & {\\rm if~|a| \\leq \\delta} \\\\
            \\delta (|a| - \\frac{1}{2} \\delta) & {\\rm otherwise,}
            \\end{array} \\right.

    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, :math:`K`).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, :math:`K`).
        delta (float): Constant variable for huber loss function
            as used in definition.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            huber loss :math:`L_{\\delta}`.

    See:
        `Huber loss - Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`_.

    """
    return HuberLoss(delta=delta, n_batch_axes=n_batch_axes)(x, t)
