import numpy

from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class AbsoluteError(function_node.FunctionNode):

    """Absolute error function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        return utils.force_array(abs(self.diff), dtype=x0.dtype),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        gx = gy * cuda.get_array_module(gy).sign(self.diff)
        return gx, -gx


def absolute_error(x0, x1):
    """Absolute error function.

    This function computes absolute error between two variables.

    """
    return AbsoluteError().apply((x0, x1))[0]
