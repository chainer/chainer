from chainer import cuda
from chainer import function
from chainer.utils import type_check
import six


class Flip(function.Function):
    """Flip an input variable in reverse order along the given axis."""

    def __init__(self, axis):
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be int')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(x_type.ndim > 0)
        if self.axis >= 0:
            type_check.expect(x_type.ndim > self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis)

    def forward(self, inputs):
        self.retain_inputs(())
        xp = cuda.get_array_module(*inputs)
        return xp.flip(inputs[0], self.axis),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*grads)
        return xp.flip(grads[0], self.axis),


def flip(x, axis):
    """Flip an input variable in reverse order along the given axis.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable.
        axis (int): Axis along which the input variable is reversed.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Flip(axis)(x)
