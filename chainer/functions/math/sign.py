import chainer
from chainer import backend
from chainer import utils


def sign(x):
    """Elementwise sign function.

    For a given input :math:`x`, this function returns :math:`sgn(x)`
    defined as

    .. math::

        sgn(x) = \\left \\{ \\begin{array}{cc}
        -1 & {\\rm if~x < 0} \\\\
        0 & {\\rm if~x = 0} \\\\
        1 & {\\rm if~x > 0} \\\\
        \\end{array} \\right.

    .. note::

        The gradient of this function is ``None`` everywhere and therefore
        unchains the computational graph.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable for which the sign is computed.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if isinstance(x, chainer.variable.Variable):
        x = x.array
    xp = backend.get_array_module(x)
    return chainer.as_variable(utils.force_array(xp.sign(x)))
