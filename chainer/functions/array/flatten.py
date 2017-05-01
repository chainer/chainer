from chainer import function


class Flatten(function.Function):

    """Flatten function."""

    def forward(self, inputs):
        return inputs[0].ravel(),

    def backward(self, inputs, grads):
        return grads[0].reshape(inputs[0].shape),


def flatten(x):
    """Flatten a given array into one dimension.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable flatten to one dimension.

    .. admonition:: Example

        >>> x = np.array([[1, 2], [3, 4]])
        >>> x.shape
        (2, 2)
        >>> y = F.flatten(x)
        >>> y.shape
        (4,)
        >>> y.data
        array([1, 2, 3, 4])

        >>> x = np.arange(8).reshape(2, 2, 2)
        >>> x.shape
        (2, 2, 2)
        >>> y = F.flatten(x)
        >>> y.shape
        (8,)
        >>> y.data
        array([0, 1, 2, 3, 4, 5, 6, 7])

    """
    return Flatten()(x)
