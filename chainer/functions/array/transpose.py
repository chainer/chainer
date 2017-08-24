import numpy

from chainer import function_node
from chainer.utils import type_check


class Transpose(function_node.FunctionNode):
    """Permute the dimensions of an array."""

    def __init__(self, axes=None):
        self.axes = axes

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @property
    def label(self):
        return 'Transpose'

    def forward(self, inputs):
        x = inputs[0]
        y = x.transpose(self.axes)
        return y,

    def backward(self, indexes, grad_outputs):
        inv_axes = self.axes
        if inv_axes:
            axes_len = len(inv_axes)
            inv_axes = tuple(numpy.argsort([ax % axes_len for ax in inv_axes]))
        return Transpose(inv_axes).apply(grad_outputs)


def transpose(x, axes=None):
    """Permute the dimensions of an input variable without copy.

    Args:
        x (~chainer.Variable): Input variable.
        axes (tuple of ints): By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

    Returns:
        ~chainer.Variable: Variable whose axes are permuted.

    """
    return Transpose(axes).apply((x,))[0]
