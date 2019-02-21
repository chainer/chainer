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

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    Args:
        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs.

    """
    return MeanSquaredError().apply((x0, x1))[0]
