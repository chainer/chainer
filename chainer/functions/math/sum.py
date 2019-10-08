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

    def __init__(self, axis=None, keepdims=False, dtype=None):
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
        self.dtype = dtype

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
        # TODO(niboshi): Support dtype argument in chainerx.sum.
        if not (self.dtype is None or self.dtype == x.dtype):
            return chainer.Fallback
        return chainerx.sum(x, axis=self.axis, keepdims=self.keepdims),

    def forward(self, inputs):
        x, = inputs
        ret = x.sum(axis=self.axis, dtype=self.dtype, keepdims=self.keepdims)
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
        if gy.dtype != self.inputs[0].dtype:
            gy = chainer.functions.cast(gy, self.inputs[0].dtype)
        return chainer.functions.broadcast_to(gy, self.inputs[0].shape),


def sum(x, axis=None, keepdims=False, dtype=None):
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
    if dtype is not None and numpy.dtype(dtype).kind != 'f':
        raise ValueError('Target dtype of F.sum must be of float kind.')

    y, = Sum(axis, keepdims, dtype).apply((x,))
    return y


class SumTo(function_node.FunctionNode):

    """Sum axes to output an array of a given shape."""

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    def forward(self, inputs):
        x, = inputs
        return utils.sum_to(x, self._shape, self._dtype),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        x_node, = self.inputs
        if gy.dtype != self.inputs[0].dtype:
            gy = chainer.functions.cast(gy, self.inputs[0].dtype)
        return chainer.functions.broadcast_to(gy, x_node.shape),


def sum_to(x, shape, dtype=None):
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
    if dtype is not None and numpy.dtype(dtype).kind != 'f':
        raise ValueError('Target dtype of F.sum_to must be of float kind.')

    if x.shape == shape:
        if dtype is None or dtype == x.dtype:
            return chainer.as_variable(x)
        return chainer.functions.cast(x, dtype)
    y, = SumTo(shape, dtype).apply((x,))
    return y
