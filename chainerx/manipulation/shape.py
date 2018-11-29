# TODO(sonots): Implement in C++
def ravel(a):
    """Returns a flattened array.

    It tries to return a view if possible, otherwise returns a copy.

    Args:
        a (~chainerx.ndarray): Array to be flattened.

    Returns:
        ~chainerx.ndarray: A flattened view of ``a`` if possible,
        otherwise a copy.

    Note:
        During backpropagation, this function propagates the gradient
        of the output array to the input array ``a``.

    .. seealso:: :func:`numpy.ravel`

    """
    return a.reshape((a.size,))
