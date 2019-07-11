from chainer import backend
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class LayerNormalization(function_node.FunctionNode):

    """Layer normalization"""

    def __init__(self, eps=1e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 2,
            gamma_type.ndim == 1,
            beta_type.ndim == 1,
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def _compute(self, xp, x):
        # xp: numpy, cupy, or chainer.functions
        mu = xp.mean(x, axis=1, keepdims=True)
        x_mu = x - mu
        squ_x_mu = xp.square(x_mu)
        var = xp.mean(squ_x_mu, axis=1, keepdims=True)
        std = xp.sqrt(var + self.eps)
        inv_std = 1. / std
        x_hat = x_mu * inv_std
        return x_mu, var, inv_std, x_hat

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        x, gamma, beta = inputs
        x_mu, var, inv_std, x_hat = self._compute(xp, x)
        scaled_x = x_hat * gamma[None, ]
        shifted_x = scaled_x + beta[None, ]
        return shifted_x,

    def backward(self, indexes, grad_outputs):
        F = chainer.functions
        x, gamma = self.get_retained_inputs()
        gy, = grad_outputs

        x_mu, var, inv_std, x_hat = self._compute(F, x)

        g_beta = F.sum(gy, axis=0)
        g_scaled_x = gy

        g_gamma = F.sum(g_scaled_x * x_hat, axis=0)
        g_x_hat = g_scaled_x * gamma

        g_inv_std = F.sum(g_x_hat * x_mu, axis=1, keepdims=True)
        g_x_mu_1 = g_x_hat * inv_std

        g_std = g_inv_std * (- 1. / (var + self.eps))
        g_var = g_std * 0.5 * inv_std

        n_units = x.shape[1]
        g_squ_x_mu = g_var * (1. / n_units)
        g_x_mu_2 = g_squ_x_mu * 2 * x_mu

        g_x_1 = g_x_mu_1 + g_x_mu_2
        g_mu = F.sum(g_x_1, axis=1, keepdims=True) * (- 1.)

        g_x_2 = g_mu * (1. / n_units)

        g_x = g_x_1 + g_x_2

        return g_x, g_gamma, g_beta,


def layer_normalization(x, gamma, beta, eps=1e-5):
    """Layer normalization.

    This function implements a "layer normalization"
    which normalizes the input units by statistics
    that are computed along the second axis,
    scales and shifts them.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Batch vectors.
            Shape of this value must be `(batch_size, unit_size)`,
            e.g., the output of :func:`~chainer.functions.linear`.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`): Scaling vectors.
        beta (:class:`~chainer.Variable` or :ref:`ndarray`): Shifting vectors.

    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_

    .. seealso::

        :class:`~chainer.links.LayerNormalization` to manage the model
        parameters ``gamma`` and ``beta``.

    """
    return LayerNormalization(eps).apply((x, gamma, beta))[0]
