import chainer
from chainer import backend
from chainer import utils


def ceil(x):
    """Elementwise ceil function.

    .. math::
       y_i = \\lceil x_i \\rceil

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if isinstance(x, chainer.variable.Variable):
        x = x.data
    xp = backend.get_array_module(x)
    return chainer.as_variable(utils.force_array(xp.ceil(x), x.dtype))
