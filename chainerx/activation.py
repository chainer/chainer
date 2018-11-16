import chainerx


def relu(a):
    """Rectified Linear Unit function.

Args:
    a (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\max (0, x)`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.
"""
    return chainerx.maximum(0, a)
