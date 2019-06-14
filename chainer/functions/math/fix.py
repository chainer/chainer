import chainer
from chainer import backend
from chainer import utils


def fix(x):
    """Elementwise fix function.

    .. math::
       y_i = \\lfix x_i \\rfix

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(x, chainer.variable.Variable):
        x = x.array
    xp = backend.get_array_module(x)
    return chainer.as_variable(utils.force_array(xp.fix(x), x.dtype))
