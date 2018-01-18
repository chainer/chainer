from chainer.functions.array import reshape
from chainer.functions.array import split_axis


def separate(x, axis=0):
    """Separates an array along a given axis.

    This function separates an array along a given axis. For example, shape of
    an array is ``(2, 3, 4)``. When it separates the array with ``axis=1``, it
    returns three ``(2, 4)`` arrays.

    This function is an inverse of :func:`chainer.functions.stack`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable to be separated.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axis (int): Axis along which variables are separated.

    Returns:
        tuple of chainer.Variable: Output variables.

    .. seealso:: :func:`chainer.functions.stack`

    .. admonition:: Example

        >>> x = np.arange(6).reshape((2, 3)).astype('f')
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
        >>> y[0].data
        array([0., 1., 2.], dtype=float32)
        >>> y = F.separate(x, axis=1)
        >>> len(y)
        3
        >>> y[0].shape
        (2,)
        >>> y[0].data
        array([0., 3.], dtype=float32)

    """
    shape = list(x.shape)
    del shape[axis]
    ys = split_axis.split_axis(x, x.shape[axis], axis, force_tuple=True)
    return tuple(reshape.reshape(y, shape) for y in ys)
