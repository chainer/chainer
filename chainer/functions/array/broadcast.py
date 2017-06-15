import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _backward_one(xp, shape, dtype, g):
    if g is None:
        return xp.zeros(shape, dtype)

    ndim = len(shape)
    if g.ndim != ndim:
        g = g.sum(axis=tuple(range(g.ndim - ndim)))
        # An input variable is always an array, not a scalar.
        # We need to convert a scalar value to a zero-dim array.
        if xp.isscalar(g):
            g = xp.array(g)

    axis = tuple(i for i, sx in enumerate(shape) if sx == 1)
    if len(axis) > 0:
        return g.sum(keepdims=True, axis=axis)
    else:
        return g


class Broadcast(function.Function):

    """Function that broadcasts given arrays."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        shapes = [type_check.eval(t).shape for t in in_types]
        r_shapes = [s[::-1] for s in shapes]
        r_filled = six.moves.zip_longest(*r_shapes, fillvalue=1)
        for ss in r_filled:
            d = max(ss)
            if not all(s == d or s == 1 for s in ss):
                expect = 'each dimension has the same size or is 1'
                actual = 'shapes: ' + ', '.join(map(str, shapes))
                raise type_check.InvalidType(expect, actual)

    def forward(self, xs):
        self.retain_inputs(())
        self._xp = cuda.get_array_module(*xs)
        self._in_shapes = [x.shape for x in xs]
        self._in_dtypes = [x.dtype for x in xs]
        return tuple(self._xp.broadcast_arrays(*xs))

    def backward(self, xs, grads):
        return tuple(
            _backward_one(self._xp, shape, dtype, g)
            for shape, dtype, g in six.moves.zip(
                self._in_shapes, self._in_dtypes, grads))


def broadcast(*args):
    """Broadcast given variables.

    Args:
        args (:class:`~chainer.Variable` or :class:`numpy.ndarray` \
        or :class:`cupy.ndarray`):
            Input variables to be broadcasted. Each dimension of the shapes \
            of the input variables must have the same size.

    Returns:
        ~chainer.Variable: :class:`~chainer.Variable` or tuple of \
            :class:`~chainer.Variable` objects which are broadcasted \
            from given arguments.

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 2)).astype('f')
        >>> y = F.broadcast(x)
        >>> np.all(x == y.data)
        True
        >>> z = np.random.uniform(0, 1, (3, 2)).astype('f')
        >>> y, w = F.broadcast(x, z)
        >>> np.all(x == y.data) & np.all(z == w.data)
        True

    """
    return Broadcast()(*args)


class BroadcastTo(function.Function):

    """Function that broadcasts an array to a new shape."""

    def __init__(self, shape):
        self._shape = tuple(shape)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

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

    def forward(self, xs):
        self.retain_inputs(())
        xp = cuda.get_array_module(*xs)
        x = xs[0]
        self._xp = xp
        self._in_shape = x.shape
        self._in_dtype = x.dtype
        if hasattr(xp, 'broadcast_to'):
            return xp.broadcast_to(x, self._shape),
        else:
            # numpy 1.9 doesn't support broadcast_to method
            dummy = xp.empty(self._shape)
            bx, _ = xp.broadcast_arrays(x, dummy)
            return bx,

    def backward(self, xs, grads):
        return _backward_one(
            self._xp, self._in_shape, self._in_dtype, grads[0]),


def broadcast_to(x, shape):
    """Broadcast a given variable to a given shape.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable be broadcasted. A \
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
        >>> y.data
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])

    """
    return BroadcastTo(shape)(x)
