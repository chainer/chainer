import chainer
from chainer.backends import cuda
from chainer import utils


def fix(x):
    """Elementwise fix function.

    .. math::
       y_i = \\lfix x_i \\rfix

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(x, chainer.variable.Variable):
        x = x.data
    xp = cuda.get_array_module(x)
    return chainer.as_variable(utils.force_array(xp.fix(x), x.dtype))
