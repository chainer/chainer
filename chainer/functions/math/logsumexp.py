import six

import chainer
from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx


class LogSumExp(function_node.FunctionNode):

    def __init__(self, axis=None):
        if axis is None:
            self.axis = None
        elif isinstance(axis, six.integer_types):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(
                isinstance(a, six.integer_types) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

        if self.axis is not None:
            for axis in self.axis:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )

    def forward_chainerx(self, inputs):
        return chainerx.logsumexp(inputs[0], self.axis),

    def forward(self, inputs):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        xp = backend.get_array_module(*inputs)

        x, = inputs
        m = x.max(axis=self.axis, keepdims=True)
        y = utils.force_array(x - m)
        xp.exp(y, out=y)
        y_sum = y.sum(axis=self.axis)
        y = xp.asarray(xp.log(y_sum) + m.reshape(y_sum.shape))
        return y,

    def backward(self, indexes, grads):
        x, = self.get_retained_inputs()
        y, = self.get_retained_outputs()
        gy, = grads

        if self.axis is not None:
            actual_axis = []
            for axis in self.axis:
                if axis < 0:
                    axis = len(x.shape) + axis
                actual_axis.append(axis)
            for axis in sorted(actual_axis):
                gy = chainer.functions.expand_dims(gy, axis=axis)
                y = chainer.functions.expand_dims(y, axis=axis)
        gy = chainer.functions.broadcast_to(gy, x.shape)
        y = chainer.functions.broadcast_to(y, x.shape)
        gx = gy * chainer.functions.exp(x - y)
        return gx,


def logsumexp(x, axis=None):
    """Log-sum-exp of array elements over a given axis.

    This function calculates logarithm of sum of exponential of array elements.

    .. math::

       y_i = \\log\\left(\\sum_j \\exp(x_{ij})\\right)

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Elements to log-sum-exp.
        axis (None, int, or tuple of int): Axis which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return LogSumExp(axis).apply((x,))[0]
