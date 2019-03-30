import six

from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class Standardize(function_node.FunctionNode):

    """Standardization for `Weight standardization
        <https://arxiv.org/abs/1903.10520>`_"""

    def __init__(self, eps=1e-5):
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
        axes = tuple(six.moves.range(1, len(x.shape)))
        mu = xp.mean(x, axis=axes, keepdims=True)
        x_mu = x - mu
        squ_x_mu = xp.square(x_mu)
        var = xp.mean(squ_x_mu, axis=axes, keepdims=True)
        std = xp.sqrt(var + self.eps)
        inv_std = 1. / std
        x_hat = x_mu * inv_std
        return x_mu, var, inv_std, x_hat

    def forward(self, inputs):
        self.retain_inputs((0,))
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        x_mu, var, inv_std, x_hat = self._compute(xp, x)
        return x_hat,

    def backward(self, indexes, grad_outputs):
        F = chainer.functions
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        axes = tuple(six.moves.range(1, len(x.shape)))

        x_mu, var, inv_std, x_hat = self._compute(F, x)

        g_inv_std = F.sum(gy * x_mu, axis=axes, keepdims=True)
        g_x_mu_1 = gy * inv_std

        g_std = g_inv_std * (- 1. / var)
        g_var = g_std * 0.5 * inv_std

        n_units = x.size / x.shape[0]
        g_squ_x_mu = g_var * (1. / n_units)
        g_x_mu_2 = g_squ_x_mu * 2 * x_mu

        g_x_1 = g_x_mu_1 + g_x_mu_2
        g_mu = F.sum(g_x_1, axis=axes, keepdims=True) * (- 1.)

        g_x_2 = g_mu * (1. / n_units)

        g_x = g_x_1 + g_x_2

        return g_x,


def standardize(x, eps=1e-5):
    """Standardize for `Weight standardization
    <https://arxiv.org/abs/1903.10520>`_.

    This function implements a "weight standardization"
    which standardize the input weights by statistics
    that are computed except the first axis.


    Args:
        x (~chainer.Variable): Weight vectors.
            e.g., the input of :func:`~chainer.functions.convolution_2d`.


    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    See: `Weight Standardization <https://arxiv.org/abs/1903.10520>`_
    """
    return Standardize(eps).apply((x,))[0]
