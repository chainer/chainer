import chainer
from chainer import backend
from chainer import utils


def floor(x):
    """Elementwise floor function.

    .. math::
       y_i = \\lfloor x_i \\rfloor

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if isinstance(x, chainer.variable.Variable):
        x = x.data
    xp = backend.get_array_module(x)
    return chainer.as_variable(utils.force_array(xp.floor(x), x.dtype))
