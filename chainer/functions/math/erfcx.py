import numpy

try:
    from scipy import special
    available_cpu = True
except ImportError as e:
    available_cpu = False
    _import_error = e

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Erfcx(function_node.FunctionNode):

    @property
    def label(self):
        return 'erfcx'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation'
                              ' of erfcx in CPU cannot be done. ' +
                              str(_import_error))
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return utils.force_array(special.erfcx(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = erfcx(x)',
            'elementwise_erfcx',
        )(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        y = self.get_retained_outputs()[0]
        return 2 * (x * y - numpy.pi ** -0.5) * gy[0],


def erfcx(x):
    """Elementwise scaled complementary error function.

    .. note::
       Forward computation in CPU cannot be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Erfcx().apply((x,))[0]
