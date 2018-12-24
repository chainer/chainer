from chainer import backend
from chainer import function_node
from chainer.functions.array import stack
from chainer.utils import type_check


class Separate(function_node.FunctionNode):

    """Function that separates a given array."""

    def __init__(self, axis):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        if self.axis >= 0:
            type_check.expect(self.axis < x_type.ndim)
        else:
            type_check.expect(-self.axis <= x_type.ndim)

    def forward(self, inputs):
        x, = inputs
        self._xp = backend.get_array_module(x)
        xs = self._xp.split(x, x.shape[self.axis], self.axis)
        ys = [self._xp.squeeze(y, self.axis) for y in xs]
        self._shape = ys[0].shape
        self._dtype = x.dtype
        return tuple(ys)

    def backward(self, indexes, grad_outputs):
        grad_outputs = [
            self._xp.zeros(self._shape, dtype=self._dtype)
            if g is None else g for g in grad_outputs]
        return stack.stack(grad_outputs, self.axis),


def separate(x, axis=0):
    """Separates an array along a given axis.

    This function separates an array along a given axis. For example, shape of
    an array is ``(2, 3, 4)``. When it separates the array with ``axis=1``, it
    returns three ``(2, 4)`` arrays.

    This function is an inverse of :func:`chainer.functions.stack`.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable to be separated.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axis (int): Axis along which variables are separated.

    Returns:
        tuple of chainer.Variable: Output variables.

    .. seealso:: :func:`chainer.functions.stack`

    .. admonition:: Example

        >>> x = np.arange(6).reshape((2, 3)).astype(np.float32)
        >>> x
        array([[0., 1., 2.],
               [3., 4., 5.]], dtype=float32)
        >>> x.shape
        (2, 3)
        >>> y = F.separate(x) # split along axis=0
        >>> isinstance(y, tuple)
        True
        >>> len(y)
        2
        >>> y[0].shape
        (3,)
        >>> y[0].array
        array([0., 1., 2.], dtype=float32)
        >>> y = F.separate(x, axis=1)
        >>> len(y)
        3
        >>> y[0].shape
        (2,)
        >>> y[0].array
        array([0., 3.], dtype=float32)

    """
    return Separate(axis).apply((x,))
