import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class MeanAbsoluteError(function_node.FunctionNode):

    """Mean absolute error function."""

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
        gx0 = coeff * cuda.get_array_module(gy.data).sign(self.diff)
        return gx0, -gx0


def mean_absolute_error(x0, x1):
    """Mean absolute error function.

    This function computes mean absolute error between two variables. The mean
    is taken over the minibatch.

    """
    return MeanAbsoluteError().apply((x0, x1))[0]
