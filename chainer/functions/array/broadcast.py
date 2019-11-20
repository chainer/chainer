import six

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check
import chainerx


class Broadcast(function_node.FunctionNode):

    """Function that broadcasts given arrays."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        shapes = [t.shape for t in in_types]
        type_check.expect_broadcast_shapes(*shapes)

    def forward(self, inputs):
        self._xp = backend.get_array_module(*inputs)
        self._in_shapes = [x.shape for x in inputs]
        self._in_dtypes = [x.dtype for x in inputs]
        return tuple(self._xp.broadcast_arrays(*inputs))

    def backward(self, indexes, grad_outputs):
        return tuple([None if grad_outputs[i] is None else
                      chainer.functions.sum_to(
                          grad_outputs[i], self.inputs[i].shape)
                      for i in indexes])


def broadcast(*args):
    """Broadcast given variables.

    Args:
        args (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be broadcasted. Each dimension of the shapes \
            of the input variables must have the same size.

    Returns:
        ~chainer.Variable: :class:`~chainer.Variable` or tuple of \
            :class:`~chainer.Variable` objects which are broadcasted \
            from the given arguments.

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 2)).astype(np.float32)
        >>> y = F.broadcast(x)
        >>> np.all(x == y.array)
        True
        >>> z = np.random.uniform(0, 1, (3, 2)).astype(np.float32)
        >>> y, w = F.broadcast(x, z)
        >>> np.all(x == y.array) & np.all(z == w.array)
        True

    """
    if len(args) == 1:
        return chainer.as_variable(args[0])
    return Broadcast().apply(args)


class BroadcastTo(function_node.FunctionNode):

    """Function that broadcasts an array to a new shape."""

    def __init__(self, shape):
        self._shape = tuple(shape)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

        ndim = type_check.make_variable(len(self._shape), 'len(shape)')
        type_check.expect(in_types[0].ndim <= ndim)

        shape = type_check.eval(in_types[0].shape)
        # check the shape in inverse order
        for i in six.moves.range(-1, -len(shape) - 1, -1):
            if shape[i] == self._shape[i] or shape[i] == 1:
                continue
            expect = 'in_type[0].shape[%d] == %d' % (i, self._shape[i])
            if self._shape[i] != 1:
                expect += ' or in_type[0].shape[%d] == 1' % i
            actual = 'in_type[0].shape: %s' % str(shape)
            raise type_check.InvalidType(expect, actual)

    def broadcast_to(self, inputs):
        x, = inputs
        return chainerx.broadcast_to(x, self.shape),

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        if hasattr(xp, 'broadcast_to'):
            return xp.broadcast_to(x, self._shape),
        else:
            # numpy 1.9 doesn't support broadcast_to method
            dummy = xp.empty(self._shape)
            bx, _ = xp.broadcast_arrays(x, dummy)
            return bx,

    def backward(self, indexes, grad_outputs):
        gx, = grad_outputs
        x_node, = self.inputs
        return chainer.functions.sum_to(gx, x_node.shape),


def broadcast_to(x, shape):
    """Broadcast a given variable to a given shape.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable to be broadcasted. A \
            :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        shape (tuple): Tuple of :class:`int` of the shape of the \
            output variable.

    Returns:
        ~chainer.Variable: Output variable broadcasted to the given shape.

    .. admonition:: Example

        >>> x = np.arange(0, 3)
        >>> x
        array([0, 1, 2])
        >>> y = F.broadcast_to(x, (3, 3))
        >>> y.array
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])

    """
    if x.shape == shape:
        return chainer.as_variable(x)

    y, = BroadcastTo(shape).apply((x,))
    return y
