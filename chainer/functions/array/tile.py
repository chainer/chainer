import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Tile(function.Function):
    """Tiling of an array."""

    def __init__(self, reps):
        if isinstance(reps, six.integer_types):
            self.reps = (reps,)
        elif isinstance(reps, tuple) and all(
                isinstance(x, six.integer_types) for x in reps):
            self.reps = reps
        else:
            raise TypeError('reps must be int or tuple of ints')

        if not all(x >= 0 for x in self.reps):
            raise ValueError('all elements in reps must be zero or larger')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        self.retain_inputs(())
        self._in_shape = inputs[0].shape
        self._in_dtype = inputs[0].dtype
        xp = cuda.get_array_module(*inputs)
        return xp.tile(inputs[0], self.reps),

    def backward(self, inputs, grads):
        reps = self.reps
        shape = tuple(self._in_shape)
        ndim = len(shape)

        # Ensure input and reps have the same length.
        if ndim > len(reps):
            reps = (1,) * (ndim - len(reps)) + reps
        elif ndim < len(reps):
            shape = (1,) * (len(reps) - ndim) + shape

        if grads[0].shape == ():
            # This case should be treated differently because numpy.num would
            # return a scalar (even if keepdims=True).
            return grads[0],

        # Reshape so that base axis and reps axis can be distinguished.
        new_shape = []
        for i in range(grads[0].ndim):
            new_shape.append(reps[i])
            new_shape.append(shape[i])
        new_shape = tuple(new_shape)

        # Sum along reps axis
        reps_axis = tuple(range(0, 2 * grads[0].ndim, 2))
        gy = grads[0].reshape(new_shape).sum(axis=reps_axis)

        if ndim < len(reps):
            return gy.reshape(self._in_shape),
        else:
            return gy,


def tile(x, reps):
    """Construct an array by tiling a given array.


    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. Let the length of ``reps`` be ``d``. If
            ``x.ndim < d``, ``x`` is treated as ``d``-dimensional array by
            prepending new axes. For example, when the shape of ``x`` is
            ``(2,)`` and tiled with 2-dim repetitions, ``x`` is treated as the
            shape ``(1, 2)``. If ``x.ndim > d``, ``reps`` is treated as
            ``x.ndim``-dimensional by pre-pending 1's. For example, when the
            shape of ``x`` is ``(2, 3, 2, 3)``, the 2-dim ``reps`` of
            ``(2, 2)`` is treated as ``(1, 1, 2, 2)``.
        reps (:class:`int` or :class:`tuple` of :class:`int` s):
            The number of times which ``x`` is replicated along each axis.

    Returns:
        ~chainer.Variable: The tiled output Variable.
        Let the length of ``reps`` be ``d``, the output has the dimension of
        ``max(d, x.ndim)``.

    .. admonition:: Example

        >>> x = np.array([0, 1, 2])
        >>> x.shape
        (3,)
        >>> y = np.tile(x, 2)
        >>> y.shape
        (6,)
        >>> y
        array([0, 1, 2, 0, 1, 2])
        >>> y = np.tile(x, (2, 2))
        >>> y.shape
        (2, 6)
        >>> y
        array([[0, 1, 2, 0, 1, 2],
               [0, 1, 2, 0, 1, 2]])
        >>> y = np.tile(x, (2, 1, 2))
        >>> y.shape
        (2, 1, 6)
        >>> y
        array([[[0, 1, 2, 0, 1, 2]],
        <BLANKLINE>
               [[0, 1, 2, 0, 1, 2]]])

        >>> x = np.array([[1, 2], [3, 4]])
        >>> x.shape
        (2, 2)
        >>> y = np.tile(x, 2)
        >>> y.shape
        (2, 4)
        >>> y
        array([[1, 2, 1, 2],
               [3, 4, 3, 4]])
        >>> y = np.tile(x, (2, 2))
        >>> y.shape
        (4, 4)
        >>> y
        array([[1, 2, 1, 2],
               [3, 4, 3, 4],
               [1, 2, 1, 2],
               [3, 4, 3, 4]])
        >>> y = np.tile(x, (2, 1, 2))
        >>> y.shape
        (2, 2, 4)
        >>> y
        array([[[1, 2, 1, 2],
                [3, 4, 3, 4]],
        <BLANKLINE>
               [[1, 2, 1, 2],
                [3, 4, 3, 4]]])

    """
    return Tile(reps)(x)
