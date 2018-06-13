import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class Diagonal(function_node.FunctionNode):

    def __init__(self, offset, axis1, axis2):
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
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
        xp = cuda.get_array_module(x)
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
    """

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable to be sliced.
        slices (int, slice, Ellipsis, None, integer array-like, boolean\
        array-like or tuple of them):
            An object to specify the selection of elements.

    Returns:
        A :class:`~chainer.Variable` object which contains sliced array of
        ``x``.
    """
    return Diagonal(offset, axis1, axis2).apply((x,))[0]
