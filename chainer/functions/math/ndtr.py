import math
import warnings

import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.functions.math import exponential
from chainer import utils
from chainer.utils import type_check


_ndtr_cpu = None


class Ndtr(function_node.FunctionNode):

    @property
    def label(self):
        return 'ndtr'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _ndtr_cpu
        if _ndtr_cpu is None:
            try:
                from scipy import special
                _ndtr_cpu = special.ndtr
            except ImportError:
                warnings.warn(
                    "SciPy is not available. Forward computation of ndtr in"
                    " CPU can be slow without SciPy.")
                _ndtr_cpu = numpy.vectorize(math.ndtr)
        self.retain_inputs((0,))
        return utils.force_array(_ndtr_cpu(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = normcdf(x)',
            'elementwise_ndtr',
        )(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return (2 * numpy.pi) ** -0.5 * exponential.exp(-0.5 * x ** 2) * gy[0],


def ndtr(x):
    """Cumulative distribution function of normal distribution.

    .. note::
       Forward computation in CPU can be slow if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Ndtr().apply((x,))[0]
