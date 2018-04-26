import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_erfinv_cpu = None


class ErfInv(function_node.FunctionNode):

    @property
    def label(self):
        return 'erfinv'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _erfinv_cpu
        if _erfinv_cpu is None:
            try:
                from scipy import special
                _erfinv_cpu = special.erfinv
            except ImportError:
                raise ImportError("SciPy is not available. Forward computation"
                                  " of erfinv in CPU can not be done.")
        self.retain_inputs((0,))
        return utils.force_array(_erfinv_cpu(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = erfinv(x)',
            'elementwise_erfinv',
        )(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return numpy.pi ** 0.5 / 2 * chainer.functions.exp(erfinv(x) ** 2) \
            * gy[0],


def erfinv(x):
    """Elementwise error function.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return ErfInv().apply((x,))[0]
