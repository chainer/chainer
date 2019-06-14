import six

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Tile(function_node.FunctionNode):

    """Tiling of an array."""

    def __init__(self, reps):
        if isinstance(reps, six.integer_types):
            self.reps = (reps,)
        elif isinstance(reps, tuple) and all(
                isinstance(x, six.integer_types) for x in reps):
            self.reps = reps
        else:
            msg = 'reps must be int or tuple of ints. \n' \
                  'Actual: {0}'.format(type(reps))
            raise TypeError(msg)

        if not all(x >= 0 for x in self.reps):
            raise ValueError('All elements in reps must be zero or larger')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        self._in_shape = inputs[0].shape
        xp = backend.get_array_module(*inputs)
        return xp.tile(inputs[0], self.reps),

    def backward(self, indexes, grad_outputs):
        reps = self.reps
        shape = tuple(self._in_shape)
        ndim = len(shape)

        # Ensure input and reps have the same length.
        if ndim > len(reps):
            reps = (1,) * (ndim - len(reps)) + reps
        elif ndim < len(reps):
            shape = (1,) * (len(reps) - ndim) + shape

        gy, = grad_outputs

        # Reshape so that base axis and reps axis can be distinguished.
        new_shape = []
        for i in range(gy.ndim):
            new_shape.append(reps[i])
            new_shape.append(shape[i])
        new_shape = tuple(new_shape)

        # Sum along reps axis
        reps_axis = tuple(range(0, 2 * gy.ndim, 2))
        gy = gy.reshape(new_shape)
        gy = chainer.functions.sum(gy, axis=reps_axis)

        if ndim < len(reps):
            return gy.reshape(self._in_shape),
        else:
            return gy,


def tile(x, reps):
    """Construct an array by tiling a given array.


    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
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
        >>> y = F.tile(x, 2)
        >>> y.shape
        (6,)
        >>> y.array
        array([0, 1, 2, 0, 1, 2])
        >>> y = F.tile(x, (2, 2))
        >>> y.shape
        (2, 6)
        >>> y.array
        array([[0, 1, 2, 0, 1, 2],
               [0, 1, 2, 0, 1, 2]])
        >>> y = F.tile(x, (2, 1, 2))
        >>> y.shape
        (2, 1, 6)
        >>> y.array
        array([[[0, 1, 2, 0, 1, 2]],
        <BLANKLINE>
               [[0, 1, 2, 0, 1, 2]]])

        >>> x = np.array([[1, 2], [3, 4]])
        >>> x.shape
        (2, 2)
        >>> y = F.tile(x, 2)
        >>> y.shape
        (2, 4)
        >>> y.array
        array([[1, 2, 1, 2],
               [3, 4, 3, 4]])
        >>> y = F.tile(x, (2, 2))
        >>> y.shape
        (4, 4)
        >>> y.array
        array([[1, 2, 1, 2],
               [3, 4, 3, 4],
               [1, 2, 1, 2],
               [3, 4, 3, 4]])
        >>> y = F.tile(x, (2, 1, 2))
        >>> y.shape
        (2, 2, 4)
        >>> y.array
        array([[[1, 2, 1, 2],
                [3, 4, 3, 4]],
        <BLANKLINE>
               [[1, 2, 1, 2],
                [3, 4, 3, 4]]])

    """
    return Tile(reps).apply((x,))[0]
