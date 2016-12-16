def scatter_update(a, indices, v, axis=0):
    """Replaces specified elements of an array with given values.

    This function acts similarly to :func:`numpy.ndarray.__setitem__` with one
    integer array in the slices argument
    (i.e. ``x.scatter_update(idx, val, axis=1)`` behaves like
    ``x[:, idx] = val``).
    This function does not have any counterpart in NumPy.

    Similarly to array indexing, negative indices are interpreted as counting
    from the end of the array.

    ``v`` needs to be broadcastable to shape
    ``a.shape[:axis] + indices.shape + a.shape[axis+1:]``.

    Example
    -------
    >>> import cupy
    >>> a = cupy.zeros((2, 3))
    >>> i = cupy.array([1, 0])
    >>> v = cupy.array([[1., 2.], [3., 4.]])
    >>> cupy.scatter_update(a, i, v, axis=1);
    >>> a
    array([[ 2.,  1.,  0.],
           [ 4.,  3.,  0.]])

    Args:
        a (ndarray): An array that gets updated.
        indices (array-like): Indices of elements that this function takes.
        v (array-like): Values to place in ``a`` at target indices.
        axis (int): The axis along which to select indices. If negative value
            is specified, it is counted backward from the end.

    .. note::

        When there are duplicate elements in ``indices``, the index among
        them that is used to store value is undefined.

        Example
        -------
        >>> a = cupy.zeros((2,))
        >>> i = cupy.arange(10000) % 2
        >>> v = cupy.arange(10000).astype(np.float)
        >>> cupy.scatter_update(a, i, v, axis=0)
        >>> a
        array([9982. 9983.])

        On the other hand, :func:`numpy.ndarray.__setitem__` stores the value
        corresponding to the last index among the duplicate elements in
        ``indices``.

        Example
        -------
        >>> a_cpu = numpy.zeros((2,))
        >>> i_cpu = numpy.arange(10000) % 2
        >>> v_cpu = numpy.arange(10000).astype(np.float)
        >>> a_cpu[i_cpu] = v_cpu
        >>> a_cpu
        array([9998., 9999.])

    .. note::

        :func:`scatter_update` does not raise error when indices exceed size of
        axes. Instead, its wrap indices.

    .. note::

        :func:`scatter_update` acts similarly to :func:`numpy.put` when inputs
        are one dimensional. However, :func:`scatter_update` does not repeat
        ``v`` when ``v`` is shorter than ``indices``.

    """
    a.scatter_update(indices, v, axis)
