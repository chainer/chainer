import six

import chainer
from chainer import function_node
from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.math import sum as sum_mod
from chainer import utils
from chainer.utils import type_check


class Mean(function_node.FunctionNode):
    """Mean of array elements over a given axis."""

    def __init__(self, axis, keepdims):
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

        self.keepdims = keepdims

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

    # TODO(kataoka): override `forward_chainerx` if `chainerx.mean` does not
    # overflow for large float16 inputs

    def forward(self, inputs):
        x, = inputs
        if self.axis is None:
            self.multiplier = 1.0 / x.size
        else:
            divider = 1
            for axis in self.axis:
                divider *= x.shape[axis]
            self.multiplier = 1.0 / divider
        ret = utils.force_array(
            x.mean(axis=self.axis, keepdims=self.keepdims))
        return ret,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        gy *= self.multiplier
        ndim = len(self.inputs[0].shape)
        if not (ndim == 0 or self.axis is None or self.keepdims):
            actual_axis = [
                axis if axis >= 0 else axis + ndim
                for axis in self.axis]
            shape = list(gy.shape)
            for axis in sorted(actual_axis):
                shape.insert(axis, 1)
            gy = chainer.functions.reshape(gy, shape)
        return chainer.functions.broadcast_to(gy, self.inputs[0].shape),


# TODO(kataoka): consider making the function public.
def _mean(x, axis, keepdims=False):
    y, = Mean(axis, keepdims).apply((x,))
    return y


def average(x, axis=None, weights=None, keepdims=False):
    """Calculate weighted average of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Elements to sum.
        axis (None or int or tuple of int): Axis which the method is performed.
            With the default (axis = None) it performs a mean over all the
            dimensions of the input array.
        weights (None or :class:`~chainer.Variable` or :ref:`ndarray`):
            An array holding weights to calculate weighted average.
            If it is ``None``, all weights are assumed to be one.
            When ``axis`` is ``None``, ``weights`` must have the same shape
            of ``x``. And when ``axis`` is ``int``, it must be 1-D array
            satisfying ``weights.shape == (x.shape[axis],)``.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if weights is None:
        return _mean(x, axis, keepdims)
    if axis is None:
        pass
    elif isinstance(axis, tuple):
        axis = [a + x.ndim if a < 0 else a for a in axis]
        axis.sort()
        for a, b in six.moves.zip(axis, axis[1:]):
            if a == b:
                raise ValueError('duplicate value in \'axis\'')
        axis = tuple(axis)
    else:
        if axis < 0:
            axis += x.ndim
        axis = (axis,)

    if axis is not None and len(axis) > 1:
        raise ValueError(
            'tuple axis is not supported when weights is given')
    divider = sum_mod.sum(weights)
    if axis is not None:
        w_shape = [d if i in axis else 1 for i, d in enumerate(x.shape)]
        weights = broadcast.broadcast_to(
            reshape.reshape(weights, w_shape), x.shape)

    x = x * weights

    x_sum = sum_mod.sum(x, axis, keepdims)
    divider = broadcast.broadcast_to(divider, x_sum.shape)
    return x_sum / divider
