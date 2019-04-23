import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class MeanAbsoluteError(function_node.FunctionNode):

    """Mean absolute error function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return numpy.array(abs(diff).sum() / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return abs(diff).sum() / diff.dtype.type(diff.size),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        coeff = gy * gy.data.dtype.type(1. / self.diff.size)
        coeff = chainer.functions.broadcast_to(coeff, self.diff.shape)
        gx0 = coeff * backend.get_array_module(gy.data).sign(self.diff)
        return gx0, -gx0


def mean_absolute_error(x0, x1):
    """Mean absolute error function.

    The function computes the mean absolute error between two variables. The
    mean is taken over the minibatch. Args ``x0`` and ``x1`` must have the
    same dimensions. This function first calculates the absolute value
    differences between the corresponding elements in x0 and x1, and then
    returns the mean of those differences.

    Args:
        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean absolute
            error of two inputs.

    .. admonition:: Example

        1D array examples:

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> y = np.array([0, 0, 0]).astype(np.float32)
        >>> F.mean_absolute_error(x, y)
        variable(2.)
        >>> x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
        >>> y = np.array([7, 8, 9, 10, 11, 12]).astype(np.float32)
        >>> F.mean_absolute_error(x, y)
        variable(6.)

        2D array example:

        In this example, there are 4 elements, and thus 4 errors
        >>> x = np.array([[1, 2], [3, 4]]).astype(np.float32)
        >>> y = np.array([[8, 8], [8, 8]]).astype(np.float32)
        >>> F.mean_absolute_error(x, y)
        variable(5.5)

        3D array example:

        In this example, there are 8 elements, and thus 8 errors
        >>> x = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8]), (2, 2, 2))
        >>> y = np.reshape(np.array([8, 8, 8, 8, 8, 8, 8, 8]), (2, 2, 2))
        >>> x = x.astype(np.float32)
        >>> y = y.astype(np.float32)
        >>> F.mean_absolute_error(x, y)
        variable(3.5)

    """
    return MeanAbsoluteError().apply((x0, x1))[0]
