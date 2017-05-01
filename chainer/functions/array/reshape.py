from chainer import function
from chainer.utils import type_check


def _count_unknown_dims(shape):
    cnt = 0
    for dim in shape:
        cnt += dim < 0
    return cnt


class Reshape(function.Function):

    """Reshapes an input array without copy."""

    def __init__(self, shape):
        cnt = _count_unknown_dims(shape)
        assert cnt == 0 or cnt == 1

        self.shape = shape

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
        )

        x_type, = in_types

        cnt = _count_unknown_dims(self.shape)
        if cnt == 0:
            type_check.expect(
                type_check.prod(x_type.shape) == type_check.prod(self.shape))
        else:
            known_size = 1
            for s in self.shape:
                if s > 0:
                    known_size *= s
            size_var = type_check.Variable(known_size,
                                           'known_size(=%d)' % known_size)
            type_check.expect(
                type_check.prod(x_type.shape) % size_var == 0)

    def forward(self, x):
        return x[0].reshape(self.shape),

    def backward(self, x, gy):
        return gy[0].reshape(x[0].shape),


def reshape(x, shape):
    """Reshapes an input variable without copy.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        shape (:class:`tuple` of :class:`int` s):
            The **size** of shape (**size** means the number of elements) must
            be equal to that of original shape. One shape dimension can be -1.
            In this case, the value is inferred from the length of the array
            and remaining dimensions.

    Returns:
        ~chainer.Variable:
            Variable that holds a reshaped version of the input variable.

    .. seealso:: :func:`numpy.reshape`, :func:`cupy.reshape`

    .. admonition:: Example

        >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> y = F.reshape(x, (8,))
        >>> y.shape
        (8,)
        >>> y.data
        array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> y = F.reshape(x, (4,-1))
        >>> y.shape
        (4, 2)
        >>> y.data
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])

    """
    return Reshape(shape)(x)
