import math
import warnings

import numpy

import chainer
from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_erf_cpu = None


class Erf(function_node.FunctionNode):

    @property
    def label(self):
        return 'erf'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _erf_cpu
        if _erf_cpu is None:
            try:
                from scipy import special
                _erf_cpu = special.erf
            except ImportError:
                warnings.warn(
                    "SciPy is not available. Forward computation of erf in CPU"
                    " can be slow without SciPy.")
                _erf_cpu = numpy.vectorize(math.erf)
        self.retain_inputs((0,))
        return utils.force_array(_erf_cpu(x[0]), dtype=x[0].dtype),

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

    .. note::
       Forward computation in CPU can be slow if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Erf().apply((x,))[0]
