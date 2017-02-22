from cupy import core


def sort(a):
    """Returns a sorted copy of an array with a stable sorting algorithm.

    Args:
        a (cupy.ndarray): Array to be sorted.

    Returns:
        cupy.ndarray: Array of the same type and shape as ``a``.

    .. note::
       For its implementation reason, ``cupy.sort`` currently supports only
       arrays with their rank of one and does not support ``axis``, ``kind``
       and ``order`` parameters that ``numpy.sort`` does support.

    .. seealso:: :func:`numpy.sort`

    """
    ret = a.copy()
    ret.sort()
    return ret


def lexsort(keys):
    """Perform an indirect sort using an array of keys.

    Args:
        keys (cupy.ndarray): ``(k, N)`` array containing ``k`` ``(N,)``-shaped
            arrays. The ``k`` different "rows" to be sorted. The last row is
            the primary sort key.

    Returns:
        cupy.ndarray: Array of indices that sort the keys.

    .. note::
        For its implementation reason, ``cupy.lexsort`` currently supports only
        keys with their rank of one or two and does not support ``axis``
        parameter that ``numpy.lexsort`` supports.

    .. seealso:: :func:`numpy.lexsort`

    """
    return core.lexsort(keys)


def argsort(a):
    """Return the indices that would sort an array with a stable sorting.

    Args:
        a (cupy.ndarray): Array to sort.

    Returns:
        cupy.ndarray: Array of indices that sort ``a``.

    .. note::
       For its implementation reason, ``cupy.argsort`` currently supports only
       arrays with their rank of one and does not support ``axis``, ``kind``
       and ``order`` parameters that ``numpy.argsort`` supports.

    .. seealso:: :func:`numpy.argsort`

    """
    return a.argsort()


# TODO(okuta): Implement msort


# TODO(okuta): Implement sort_complex


# TODO(okuta): Implement partition


# TODO(okuta): Implement argpartition
