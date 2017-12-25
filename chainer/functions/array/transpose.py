import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
from chainer.graph_optimimzations.static_graph import static_schedule_func


class Transpose(function_node.FunctionNode):
    """Permute the dimensions of an array."""

    def __init__(self, axes=None):
        self.axes = axes

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @static_schedule_func
    def static_transpose(self, x, y):
        # todo: optimize later to prevent unnecessary allocation.
        y[:] = x.transpose(self.axes)

    @property
    def label(self):
        return 'Transpose'

    def forward(self, inputs):
        x = inputs[0]
        #y = x.transpose(self.axes)
        xp = cuda.get_array_module(x)
        # Determine the shape of y:
        if self.axes is not None:
            y_shape = tuple([x.shape[n] for n in self.axes])
        else:
            # Reverse the axes
            y_shape = x.shape[::-1]
        y = xp.empty(y_shape).astype(x.dtype)
        self.static_transpose(x, y)
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
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable to be transposed.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axes (tuple of ints): By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

    Returns:
        ~chainer.Variable: Variable whose axes are permuted.

    .. admonition:: Example

        >>> x = np.array([[[0, 1, 2], [3, 4, 5]]], 'f')
        >>> x.shape
        (1, 2, 3)
        >>> y = F.transpose(x)  # reverse the dimensions
        >>> y.shape
        (3, 2, 1)
        >>> y.data
        array([[[ 0.],
                [ 3.]],
        <BLANKLINE>
               [[ 1.],
                [ 4.]],
        <BLANKLINE>
               [[ 2.],
                [ 5.]]], dtype=float32)
        >>> y = F.transpose(x, axes=(1, 0, 2)) # swap 1st and 2nd axis
        >>> y.shape
        (2, 1, 3)
        >>> y.data
        array([[[ 0.,  1.,  2.]],
        <BLANKLINE>
               [[ 3.,  4.,  5.]]], dtype=float32)

    """
    return Transpose(axes).apply((x,))[0]
