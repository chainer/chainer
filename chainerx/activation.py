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
    # TODO(imanishi): The function should also be available to C++ users
    return chainerx.maximum(0, a)


def sigmoid(a):
    """Element-wise sigmoid logistic function.

Args:
    a (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`:
        Returned array: :math:`y = (1 + \\exp (-x))^{-1}`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.
"""
    # TODO(imanishi): The function should also be available to C++ users
    return (chainerx.tanh(a / 2.0) + 1.0) / 2.0
