import warnings

import chainer.functions
from chainer.functions.math import clip


class ClippedReLU(clip.Clip):

    """Clipped Rectifier Unit function.

    Clipped ReLU is written as
    :math:`ClippedReLU(x, z) = \\min(\\max(0, x), z)`,
    where :math:`z(>0)` is a parameter to cap return value of ReLU.

    """

    def __init__(self, z):
        warnings.warn(
            'ClippedReLU is deprecated. Please use clipped_relu instead.',
            DeprecationWarning)
        super(ClippedReLU, self).__init__(0.0, z)


def clipped_relu(x, z=20.0):
    """Clipped Rectifier Unit function.

    For a clipping value :math:`z(>0)`, it computes

    .. math:: \\text{ClippedReLU}(x, z) = \\min(\\max(0, x), z).

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.
        z (float): Clipping value. (default = 20.0)

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_n)`-shaped float array.

    .. admonition:: Example

        >>> x = np.random.uniform(-100, 100, (10, 20)).astype(np.float32)
        >>> z = 10.0
        >>> np.any(x < 0)
        True
        >>> np.any(x > z)
        True
        >>> y = F.clipped_relu(x, z=z)
        >>> np.any(y.data < 0)
        False
        >>> np.any(y.data > z)
        False

    """
    return chainer.functions.clip(x, 0.0, z)
