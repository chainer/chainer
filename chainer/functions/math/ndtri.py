try:
    from scipy import special
    available_cpu = True
except ImportError as e:
    available_cpu = False
    _import_error = e
import math

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Ndtri(function_node.FunctionNode):

    @property
    def label(self):
        return 'ndtri'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation'
                              ' of ndtri in CPU can not be done.' +
                              str(_import_error))
        self.retain_outputs((0,))
        return utils.force_array(special.ndtri(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_outputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = normcdfinv(x)',
            'elementwise_ndtri',
        )(x[0]),

    def backward(self, indexes, gy):
        y, = self.get_retained_outputs()
        sqrt_2pi = (2 * math.pi) ** 0.5
        return sqrt_2pi * chainer.functions.exp(0.5 * y ** 2) * gy[0],


def ndtri(x):
    """Elementwise inverse function of ndtr.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Ndtri().apply((x,))[0]
