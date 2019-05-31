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


BACKWORDC = math.pi ** 0.5 / 2


class ErfInv(function_node.FunctionNode):

    @property
    def label(self):
        return 'erfinv'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation'
                              ' of erfinv in CPU cannot be done. ' +
                              str(_import_error))
        self.retain_outputs((0,))
        return utils.force_array(special.erfinv(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_outputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = erfinv(x)',
            'elementwise_erfinv',
        )(x[0]),

    def backward(self, indexes, gy):
        y, = self.get_retained_outputs()
        return BACKWORDC * chainer.functions.exp(y ** 2) * gy[0],


def erfinv(x):
    """Elementwise inverse function of error function.

    .. note::
       Forward computation in CPU cannot be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return ErfInv().apply((x,))[0]
