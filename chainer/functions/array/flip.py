import six

from chainer import backend
from chainer import function_node
from chainer.utils import type_check


def _flip(array, axis):
    indices = [slice(None)] * array.ndim
    indices[axis] = slice(None, None, -1)
    return array[tuple(indices)]


class Flip(function_node.FunctionNode):
    """Flips an input variable in reverse order along the given axis."""

    def __init__(self, axis):
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be int')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]

        type_check.expect(x_type.ndim > 0)
        if self.axis >= 0:
            type_check.expect(x_type.ndim > self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis)

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        if hasattr(xp, 'flip'):  # numpy.flip is supported from version 1.12.0
            return xp.flip(inputs[0], self.axis),
        else:
            return _flip(inputs[0], self.axis),

    def backward(self, indexes, grad_outputs):
        return flip(grad_outputs[0], self.axis),


def flip(x, axis):
    """Flips an input variable in reverse order along the given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable.
        axis (int): Axis along which the input variable is reversed.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Flip(axis).apply((x,))[0]
