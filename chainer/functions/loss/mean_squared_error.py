import numpy

from chainer import function_node
from chainer.functions.array import broadcast
from chainer.utils import type_check


class MeanSquaredError(function_node.FunctionNode):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, indexes, gy):
        ret = []
        coeff = (self.diff * 2. / self.diff.size).astype(gy[0].dtype)
        gx0 = broadcast.broadcast_to(gy[0], coeff.shape) * coeff
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    Args:
        x0 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs.

    """
    return MeanSquaredError().apply((x0, x1))[0]
