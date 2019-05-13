import six

from chainer import backend
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class _SetItemZero(function_node.FunctionNode):

    """Write values to mask of zero-initialized array"""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        y = xp.zeros(self.mask.shape, x.dtype)
        y[self.mask] = x
        return y,

    def backward(self, indices, grad_outputs):
        g, = grad_outputs
        return g[self.mask],


class Standardize(function_node.FunctionNode):

    """Standardization for `Weight standardization
        <https://arxiv.org/abs/1903.10520>`_"""

    def __init__(self, eps=1e-6):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2,
        )

    def _compute(self, xp, x):
        # xp: numpy, cupy, or chainer.functions
        axes = tuple(six.moves.range(1, x.ndim))
        mu = xp.mean(x, axis=axes, keepdims=True)
        x_mu = x - mu
        squ_x_mu = xp.square(x_mu)
        var = xp.mean(squ_x_mu, axis=axes, keepdims=True)
        if xp is chainer.functions:
            std_noeps = xp.sqrt(var)
        else:
            std_noeps = xp.sqrt(var, dtype=x.dtype)
        std = std_noeps + x.dtype.type(self.eps)
        x_hat = x_mu / std
        return x_mu, std_noeps, std, x_hat, axes

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = backend.get_array_module(x)
        x_mu, std_noeps, std, x_hat, axes = self._compute(xp, x)
        return x_hat,

    def backward(self, indexes, grad_outputs):
        F = chainer.functions
        x, = self.get_retained_inputs()
        gy, = grad_outputs

        x_mu, std_noeps, std, x_hat, axes = self._compute(F, x)
        g_x_mu_1 = std * gy
        g_std = F.mean((x_mu * gy), axis=axes, keepdims=True)

        # _standardize with eps has continuous backward. However,
        # the backward is not differentiable for the indices of zero vectors.
        # To avoid nan in double backward, do not compute outside of mask.
        mask = std_noeps.array != 0
        g_var, = _SetItemZero(mask).apply((g_std[mask] / std_noeps[mask],))

        g_x_mu_2 = g_var * x_mu
        g_x_1 = (g_x_mu_1 - g_x_mu_2) / (std ** 2)
        g_mu = F.mean(g_x_1, axis=axes, keepdims=True)

        return g_x_1 - g_mu,


def _standardize(x, eps=1e-6):
    """Weight Standardization function.

    This function implements a "weight standardization"
    which standardizes the input weights by statistics
    that are computed except the first axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Weight vectors.
            e.g., the input of :func:`~chainer.functions.convolution_2d`.

    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    See: `Weight Standardization <https://arxiv.org/abs/1903.10520>`_
    """
    return Standardize(eps).apply((x,))[0]
