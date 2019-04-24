import numpy

from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class MeanSquaredError(function_node.FunctionNode):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, indexes, gy):
        x0, x1 = self.get_retained_inputs()
        ret = []
        diff = x0 - x1
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx0 = gy0 * diff * (2. / diff.size)
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret


def mean_squared_error(x0, x1):
    """Mean squared error function.

    The function computes the mean squared error between two variables. The
    mean is taken over the minibatch. Args ``x0`` and ``x1`` must have the
    same dimensions. Note that the error is not scaled by 1/2.

    Args:
        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs.

     .. admonition:: Example

        1D array examples:

        >>> x = np.array([1, 2, 3, 4]).astype(np.float32)
        >>> y = np.array([0, 0, 0, 0]).astype(np.float32)
        >>> F.mean_squared_error(x, y)
        variable(7.5)
        >>> x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
        >>> y = np.array([7, 8, 9, 10, 11, 12]).astype(np.float32)
        >>> F.mean_squared_error(x, y)
        variable(36.)

        2D array example:

        In this example, there are 4 elements, and thus 4 errors
        >>> x = np.array([[1, 2], [3, 4]]).astype(np.float32)
        >>> y = np.array([[8, 8], [8, 8]]).astype(np.float32)
        >>> F.mean_squared_error(x, y)
        variable(31.5)

        3D array example:

        In this example, there are 8 elements, and thus 8 errors
        >>> x = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8]), (2, 2, 2))
        >>> y = np.reshape(np.array([8, 8, 8, 8, 8, 8, 8, 8]), (2, 2, 2))
        >>> x = x.astype(np.float32)
        >>> y = y.astype(np.float32)
        >>> F.mean_squared_error(x, y)
        variable(17.5)

    """
    return MeanSquaredError().apply((x0, x1))[0]
