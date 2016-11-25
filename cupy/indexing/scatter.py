def scatter_update(a, indices, v, axis=0):
    """Replaces specified elements of an array with given values.

    ``v`` needs to be broadcastable to shape
        ``a.shape[:axis] + indices.shape + a.shape[axis+1:]``.

    Args:
        indices (array-like): Indices of elements that this function takes.
        v (array-like): Values to place in ``a`` at target indices.
        axis (int): The axis along which to select indices.

    .. note::
    
        When there are duplicating elements in ``indices``, the index among
            them that is used to store value is undefined.

        Examples
        --------
        >>> a = cupy.zeros((2,))
        >>> i = cupy.arange(10001) % 2
        >>> v = cupy.arange(10000).astype(np.float)
        >>> a.scatter_update(i, v, axis=0)
        >>> a
        [9982. 9983.]

    """
    a.scatter_update(indices, v, axis)
