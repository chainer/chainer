import chainerx


# TODO(sonots): Implement in C++
def clip(a, a_min, a_max):
    """Clips the values of an array to a given interval.

    Given an interval, values outside the interval are clipped to the
    interval edges. For example, if an interval of ``[0, 1]`` is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.

    Args:
        a (~chainerx.ndarray): Array containing elements to clip.
        a_min (scalar): Maximum value.
        a_max (scalar): Minimum value.

    Returns:
        ~chainerx.ndarray: An array with the elements of ``a``, but where
        values < ``a_min`` are replaced with ``a_min``,
        and those > ``a_max`` with ``a_max``.

    Note:
        The :class:`~chainerx.ndarray` typed ``a_min`` and ``a_max`` are
        not supported yet.

    Note:
        During backpropagation, this function propagates the gradient
        of the output array to the input array ``a``.

    .. seealso:: :func:`numpy.clip`

    """
    return -chainerx.maximum(-chainerx.maximum(a, a_min), -a_max)
