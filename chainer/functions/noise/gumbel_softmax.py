from chainer import backend
import chainer.functions
from chainer import variable


def gumbel_softmax(log_pi, tau=0.1, axis=1):
    """Gumbel-Softmax sampling function.

    This function draws samples :math:`y_i` from Gumbel-Softmax distribution,

    .. math::
        y_i = {\\exp((g_i + \\log\\pi_i)/\\tau)
        \\over \\sum_{j}\\exp((g_j + \\log\\pi_j)/\\tau)},

    where :math:`\\tau` is a temperature parameter and
    :math:`g_i` s are samples drawn from
    Gumbel distribution :math:`Gumbel(0, 1)`

    See `Categorical Reparameterization with Gumbel-Softmax
    <https://arxiv.org/abs/1611.01144>`_.

    Args:
        log_pi (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable
            representing pre-normalized log-probability :math:`\\log\\pi`.
        tau (:class:`~float` or :class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable representing temperature :math:`\\tau`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    xp = backend.get_array_module(log_pi)
    if log_pi.ndim < 1:
        return variable.Variable(xp.ones((), log_pi.dtype))
    dtype = log_pi.dtype
    g = xp.random.gumbel(size=log_pi.shape).astype(dtype)
    y = chainer.functions.softmax((log_pi + g) / tau, axis=axis)

    return y
