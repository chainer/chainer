import six

from chainer import backend
from chainer import function_node
from chainer.utils import type_check


def argone(iterable):
    result = []
    for i, x in enumerate(iterable):
        if not isinstance(x, six.integer_types):
            raise ValueError('elements in iterable must be int')
        if x == 1:
            result.append(i)
    return result


class Squeeze(function_node.FunctionNode):

    """Remove dimensions of size one from the shape of a ndarray."""

    def __init__(self, axis=None):

        if axis is None:
            self.axis = None
        elif isinstance(axis, six.integer_types):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(
                isinstance(x, six.integer_types) for x in axis):
            self.axis = axis
        else:
            raise TypeError('axis must be None, int or tuple of ints')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        if self.axis is not None:
            for x in self.axis:
                if x >= 0:
                    type_check.expect(x < x_type.ndim)
                else:
                    type_check.expect(-x_type.ndim <= x)

    def forward_chainerx(self, inputs):
        x, = inputs
        return x.squeeze(self.axis),

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        return xp.squeeze(x, self.axis),

    def backward(self, indexes, grad_outputs):
        if self.axis is None:
            axis = tuple(argone(self.inputs[0].shape))
        else:
            axis = self.axis
            ndim = len(self.inputs[0].shape)
            axis = [x + ndim if x < 0 else x for x in axis]
            axis.sort()
        gx, = grad_outputs

        shape = list(gx.shape)
        for x in axis:          # axis needs to be sorted
            shape.insert(x, 1)
        return gx.reshape(shape),


def squeeze(x, axis=None):
    """Remove dimensions of size one from the shape of a ndarray.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axis (None or int or tuple of ints): A subset of the single-dimensional
            entries in the shape to remove. If ``None`` is supplied, all of
            them are removed. The dimension index starts at zero. If an axis
            with dimension greater than one is selected, an error is raised.

    Returns:
        ~chainer.Variable: Variable whose dimensions of size 1 are removed.

    .. admonition:: Example

        >>> x = np.array([[[[0, 1, 2]]], [[[3, 4, 5]]]], np.float32)
        >>> x.shape
        (2, 1, 1, 3)
        >>> y = F.squeeze(x)
        >>> y.shape
        (2, 3)
        >>> y.array
        array([[0., 1., 2.],
               [3., 4., 5.]], dtype=float32)
        >>> y = F.squeeze(x, axis=1)
        >>> y.shape
        (2, 1, 3)
        >>> y.array
        array([[[0., 1., 2.]],
        <BLANKLINE>
               [[3., 4., 5.]]], dtype=float32)
        >>> y = F.squeeze(x, axis=(1, 2))
        >>> y.shape
        (2, 3)
        >>> y.array
        array([[0., 1., 2.],
               [3., 4., 5.]], dtype=float32)

    """
    y, = Squeeze(axis).apply((x,))
    return y
