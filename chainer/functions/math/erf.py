import numpy

import chainer
from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Erf(function_node.FunctionNode):

    @property
    def label(self):
        return 'erf'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        try:
            from scipy import special
        except ImportError:
            raise RuntimeError(
                "CPU computation of erf requires scipy")
        self.retain_inputs((0,))
        return utils.force_array(special.erf(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = erf(x)',
            'elementwise_erf',
        )(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return 2 / numpy.pi ** 0.5 * chainer.functions.exp(-x ** 2) * gy[0],


def erf(x):
    """Elementwise error function.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Erf().apply((x,))[0]
