import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Rollaxis(function.Function):

    """Roll axis of an array."""

    def __init__(self, axis, start):
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be int')
        if not isinstance(start, six.integer_types):
            raise TypeError('start must be int')

        self.axis = axis
        self.start = start

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        if self.axis >= 0:
            type_check.expect(x_type.ndim > self.axis)
        else:
            type_check.expect(x_type.ndim > -self.axis - 1)

        if self.start >= 0:
            type_check.expect(x_type.ndim >= self.start)
        else:
            type_check.expect(x_type.ndim > -self.start - 1)

    def forward(self, inputs):
        self.retain_inputs(())
        self._in_ndim = inputs[0].ndim
        xp = cuda.get_array_module(*inputs)
        return xp.rollaxis(inputs[0], self.axis, self.start),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*grads)
        axis = self.axis
        if axis < 0:
            axis += self._in_ndim
        start = self.start
        if start < 0:
            start += self._in_ndim

        if axis > start:
            axis += 1
        else:
            start -= 1

        return xp.rollaxis(grads[0], start, axis),


def rollaxis(x, axis, start=0):
    """Roll the axis backwards to the given position.

    Args:
        x (~chainer.Variable): Input variable.
        axis (int): The axis to roll backwards.
        start (int): The place to which the axis is moved.

    Returns:
        ~chainer.Variable: Variable whose axis is rolled.
    """
    return Rollaxis(axis, start)(x)
