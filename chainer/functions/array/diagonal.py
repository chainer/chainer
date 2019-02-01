import numpy

from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Diagonal(function_node.FunctionNode):

    def __init__(self, offset, axis1, axis2):
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        in_type = in_types[0]
        type_check.expect(max(self.axis1, self.axis2) < in_type.ndim)
        type_check.expect(-in_type.ndim <= min(self.axis1, self.axis2))

    def forward(self, inputs):
        x, = inputs
        self._in_shape = x.shape
        y = x.diagonal(offset=self.offset, axis1=self.axis1, axis2=self.axis2)
        return y,

    def backward(self, indexes, grad_outputs):
        return DiagonalGrad(
            self._in_shape, self.offset, self.axis1, self.axis2
        ).apply(grad_outputs)


class DiagonalGrad(function_node.FunctionNode):

    def __init__(self, out_shape, offset, axis1, axis2):
        self.out_shape = out_shape
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        y = xp.zeros(self.out_shape, x.dtype)
        y_diag = y.diagonal(
            offset=self.offset, axis1=self.axis1, axis2=self.axis2)
        if xp is numpy:
            y_diag.setflags(write=True)
        y_diag[...] = x
        return y,

    def backward(self, indexes, grad_outputs):
        return Diagonal(self.offset, self.axis1, self.axis2).apply(
            grad_outputs)


def diagonal(x, offset=0, axis1=0, axis2=1):
    """Take diagonal

    Axes other than ``axis1`` and ``axis2`` are regarded as batch dimensions.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            A variable to be sliced.
        offset (int): Offset from the principal diagonal. An upper diagonal
            matrix can have nonzero diagonals with nonnegative offsets.
        axis1 (int): First axis (that has row indices) of matrix
        axis2 (int): Second axis (that has column indices) of matrix

    Returns:
        ~chainer.Variable: (Batched) diagonal vectors

    .. admonition:: Example

        >>> x = chainer.Variable(np.arange(9).reshape(3, 3).astype(np.float32))
        >>> x
        variable([[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]])
        >>> chainer.functions.diagonal(x, offset=1)
        variable([1., 5.])

    """
    return Diagonal(offset, axis1, axis2).apply((x,))[0]
