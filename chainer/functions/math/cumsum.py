import six

from chainer import backend
from chainer import function_node
from chainer.functions.array import flip
from chainer.utils import type_check


class Cumsum(function_node.FunctionNode):
    """Cumulative sum of array elements over a given axis."""

    def __init__(self, axis=None):
        if isinstance(axis, six.integer_types) or axis is None:
            self.axis = axis
        else:
            raise TypeError('axis must be int or None')

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

        if self.axis is not None:
            if self.axis >= 0:
                type_check.expect(self.axis < in_types[0].ndim)
            else:
                type_check.expect(-self.axis - 1 < in_types[0].ndim)

    def forward(self, inputs):
        x, = inputs
        self._in_shape = x.shape
        xp = backend.get_array_module(x)
        return xp.cumsum(x, axis=self.axis),

    def backward(self, indexes, grad_outputs):
        gy = grad_outputs[0]
        axis = self.axis

        if axis is not None:
            gx = flip.flip(cumsum(flip.flip(gy, axis), axis), axis)
        else:
            gx = flip.flip(cumsum(flip.flip(gy, 0), 0), 0)
            gx = gx.reshape(self._in_shape)

        return gx,


def cumsum(x, axis=None):
    """Cumulative sum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Elements to calculate the cumulative sum.
        axis (int or None):
            Axis along which the cumulative sum is taken.
            If it is not specified, the input is flattened.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Cumsum(axis).apply((x,))[0]
