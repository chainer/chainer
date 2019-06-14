import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Expm1(function_node.FunctionNode):

    @property
    def label(self):
        return 'expm1'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_outputs((0,))
        return utils.force_array(numpy.expm1(x[0])),

    def forward_gpu(self, x):
        self.retain_outputs((0,))
        return cuda.cupy.expm1(x[0]),

    def backward(self, indexes, gy):
        y = self.get_retained_outputs()[0]
        return (y + 1.0) * gy[0],


def expm1(x):
    """Elementwise exponential minus one function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Expm1().apply((x,))[0]
