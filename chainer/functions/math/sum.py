import numpy

from chainer import cuda
from chainer import function_node
from chainer.functions.array import reshape
from chainer.utils import type_check


class Sum(function_node.FunctionNode):
    """Sum of array elements over a given axis."""

    keepdims = False

    def __init__(self, axis=None, keepdims=False):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(isinstance(a, int) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

        self.keepdims = keepdims

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

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

    def forward(self, inputs):
        x, = inputs
        self._in_shape = x.shape
        self._in_dtype = x.dtype
        ret = x.sum(axis=self.axis, keepdims=self.keepdims)
        if cuda.get_array_module(x) is numpy:
            ret = numpy.asarray(ret)
        return ret,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ndim = len(self._in_shape)
        if not (ndim == 0 or self.axis is None or self.keepdims):
            actual_axis = [
                axis if axis >= 0 else axis + ndim
                for axis in self.axis]
            shape = list(gy.shape)
            for axis in sorted(actual_axis):
                shape.insert(axis, 1)
            gy = reshape.reshape(gy, shape)

        # to avoid import error
        from chainer.functions.array import broadcast
        return broadcast.broadcast_to(gy, self._in_shape),


def sum(x, axis=None, keepdims=False):
    """Sum of array elements over a given axis.

    Args:
        x (~chainer.Variable): Elements to sum.
        axis (None, int, or tuple of int): Axis which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Sum(axis, keepdims).apply((x,))[0]
