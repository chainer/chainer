import chainer
from chainer import function_node
from chainer.utils import type_check


def _count_unknown_dims(shape):
    cnt = 0
    for dim in shape:
        cnt += dim < 0
    return cnt


class Reshape(function_node.FunctionNode):

    """Reshapes an input array without copy."""

    def __init__(self, shape):
        self.shape = shape
        self._cnt = _count_unknown_dims(shape)
        assert self._cnt <= 1

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        if self._cnt == 0:
            type_check.expect(
                type_check.prod(x_type.shape) == type_check.prod(self.shape))
        else:
            known_size = 1
            for s in self.shape:
                if s > 0:
                    known_size *= s
            size_var = type_check.make_variable(
                known_size, 'known_size(=%d)' % known_size)
            type_check.expect(
                type_check.prod(x_type.shape) % size_var == 0)

    def forward_chainerx(self, inputs):
        x, = inputs
        return x.reshape(self.shape),

    def forward(self, inputs):
        x, = inputs
        return x.reshape(self.shape),

    def backward(self, indexes, grad_outputs):
        gx, = grad_outputs
        return reshape(gx, self.inputs[0].shape),


def reshape(x, shape):
    """Reshapes an input variable without copy.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        shape (:class:`tuple` of :class:`int` s):
            Expected shape of the output array. The number of elements which
            the array of ``shape`` contains must be equal to that of input
            array. One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.

    Returns:
        ~chainer.Variable:
            Variable that holds a reshaped version of the input variable.

    .. seealso:: :func:`numpy.reshape`, :func:`cupy.reshape`

    .. admonition:: Example

        >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> y = F.reshape(x, (8,))
        >>> y.shape
        (8,)
        >>> y.array
        array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> y = F.reshape(x, (4, -1))  # the shape of output is inferred
        >>> y.shape
        (4, 2)
        >>> y.array
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
        >>> y = F.reshape(x, (4, 3))  \
# the shape of input and output are not consistent
        Traceback (most recent call last):
        ...
        chainer.utils.type_check.InvalidType:
        Invalid operation is performed in: Reshape (Forward)
        <BLANKLINE>
        Expect: prod(in_types[0].shape) == prod((4, 3))
        Actual: 8 != 12

    """
    if x.shape == shape:
        return chainer.as_variable(x)
    y, = Reshape(shape).apply((x,))
    return y
