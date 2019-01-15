import chainer


def flatten(x):
    """Flatten a given array into one dimension.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable flatten to one dimension.

    .. note::

        When you input a scalar array (i.e. the shape is ``()``),
        you can also get the one dimension array whose shape is ``(1,)``.

    .. admonition:: Example

        >>> x = np.array([[1, 2], [3, 4]])
        >>> x.shape
        (2, 2)
        >>> y = F.flatten(x)
        >>> y.shape
        (4,)
        >>> y.array
        array([1, 2, 3, 4])

        >>> x = np.arange(8).reshape(2, 2, 2)
        >>> x.shape
        (2, 2, 2)
        >>> y = F.flatten(x)
        >>> y.shape
        (8,)
        >>> y.array
        array([0, 1, 2, 3, 4, 5, 6, 7])

    """
    return chainer.functions.reshape(x, (x.size,))
