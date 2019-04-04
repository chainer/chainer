import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Log1p(function_node.FunctionNode):

    @property
    def label(self):
        return 'log1p'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        return utils.force_array(numpy.log1p(x[0])),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.cupy.log1p(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()
        return gy[0] / (x[0] + 1.0),


def log1p(x):
    """Elementwise natural logarithm plus one function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Log1p().apply((x,))[0]
