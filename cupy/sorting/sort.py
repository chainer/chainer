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


# TODO(okuta): Implement lexsort


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


def msort(a):
    """Returns a copy of an array sorted along the first axis.

    Args:
        a (cupy.ndarray): Array to be sorted.

    Returns:
        cupy.ndarray: Array of the same type and shape as ``a``.

    .. note:
       ``numpy.msort(a)``, the numpy counterpart of ``cupy.msort(a)``, is
       equivalent to ``numpy.sort(a, axis=0)``. For its implementation reason,
       ``cupy.sort`` currently supports only sorting an array with its rank of
       one, so ``cupy.msort(a)`` is actually the same as``cupy.sort(a)`` for
       now.

    .. seealso:: :func:`numpy.msort`

    """
    # TODO(takagi): Support axis argument.
    # TODO(takagi): Support ranks of two or more.
    # TODO(takagi): Support float16 and bool.
    return sort(a)


# TODO(okuta): Implement sort_complex


# TODO(okuta): Implement partition


# TODO(okuta): Implement argpartition
