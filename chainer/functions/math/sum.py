import numpy
import six

import chainer
from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx


class Sum(function_node.FunctionNode):
    """Sum of array elements over a given axis."""

    keepdims = False

    def __init__(self, axis=None, keepdims=False):
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

    def forward_chainerx(self, inputs):
        x, = inputs
        return chainerx.sum(x, axis=self.axis, keepdims=self.keepdims),

    def forward(self, inputs):
        x, = inputs
        ret = x.sum(axis=self.axis, keepdims=self.keepdims)
        if backend.get_array_module(x) is numpy:
            ret = numpy.asarray(ret)
        return ret,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
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


def sum(x, axis=None, keepdims=False):
    """Sum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Elements to sum.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axis (None, int, or tuple of int): Axis along which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x = np.arange(6).reshape(2,3).astype(np.float32)
        >>> x
        array([[0., 1., 2.],
               [3., 4., 5.]], dtype=float32)
        >>> y = F.sum(x)
        >>> y.shape
        ()
        >>> y.array
        array(15., dtype=float32)
        >>> y = F.sum(x, axis=1)
        >>> y.shape
        (2,)
        >>> y.array
        array([ 3., 12.], dtype=float32)
        >>> y = F.sum(x, keepdims=True)
        >>> y.shape
        (1, 1)
        >>> y.array
        array([[15.]], dtype=float32)

    """
    y, = Sum(axis, keepdims).apply((x,))
    return y


class SumTo(function_node.FunctionNode):

    """Sum axes to output an array of a given shape."""

    def __init__(self, shape):
        self._shape = shape

    def forward(self, inputs):
        x, = inputs
        return utils.sum_to(x, self._shape),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        x_node, = self.inputs
        return chainer.functions.broadcast_to(gy, x_node.shape),


def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        shape (tuple of int): The target shape.

    Returns:
        ~chainer.Variable: Output variable of shape ``shape``.

    .. admonition:: Example

        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> x
        array([[1., 2., 3.],
               [4., 5., 6.]])
        >>> y = F.sum_to(x, (1, 3))
        >>> y
        variable([[5., 7., 9.]])
        >>> z = F.sum_to(x, (2, 1))
        >>> z
        variable([[ 6.],
                  [15.]])

    """
    if x.shape == shape:
        return chainer.as_variable(x)
    y, = SumTo(shape).apply((x,))
    return y
