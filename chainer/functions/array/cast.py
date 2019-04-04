import numpy

import chainer
from chainer import function_node
from chainer.utils import type_check


class Cast(function_node.FunctionNode):

    """Cast function."""

    def __init__(self, typ):
        self.type = typ

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

    def forward(self, x):
        self._in_type = x[0].dtype.type
        return x[0].astype(self.type, copy=False),

    def backward(self, indexes, g):
        if numpy.dtype(self._in_type).kind != 'f':
            gx = None
        else:
            gx = cast(g[0], self._in_type)
        return gx,


def cast(x, typ):
    """Cast an input variable to a given type.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable to be casted. A \
            :math:`(s_1, s_2, ..., s_N)`-shaped array.
        typ (:class:`str` of dtype or :class:`numpy.dtype`):
            Typecode or data type to cast.

    Returns:
        ~chainer.Variable: Variable holding a casted array.

    .. admonition:: Example

        >>> x = np.arange(0, 3, dtype=np.float64)
        >>> x.dtype
        dtype('float64')
        >>> y = F.cast(x, np.float32)
        >>> y.dtype
        dtype('float32')
        >>> y = F.cast(x, 'float16')
        >>> y.dtype
        dtype('float16')

    """
    if x.dtype == typ:
        if not chainer.config.enable_backprop:
            return chainer.as_variable(x)
    return Cast(typ).apply((x,))[0]
