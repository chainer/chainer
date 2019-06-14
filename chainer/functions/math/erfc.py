import math
import warnings

import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_erfc_cpu = None


class Erfc(function_node.FunctionNode):

    @property
    def label(self):
        return 'erfc'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _erfc_cpu
        if _erfc_cpu is None:
            try:
                from scipy import special
                _erfc_cpu = special.erfc
            except ImportError:
                warnings.warn(
                    'SciPy is not available. Forward computation of erfc in'
                    ' CPU can be slow without SciPy.',
                    chainer.warnings.PerformanceWarning)
                _erfc_cpu = numpy.vectorize(math.erfc)
        self.retain_inputs((0,))
        return utils.force_array(_erfc_cpu(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = erfc(x)',
            'elementwise_erfc',
        )(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return -2 / numpy.pi ** 0.5 * chainer.functions.exp(-x ** 2) * gy[0],


def erfc(x):
    """Elementwise complementary error function.

    .. note::
       Forward computation in CPU can be slow if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Erfc().apply((x,))[0]
