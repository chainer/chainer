import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_digamma_cpu = None


class DiGamma(function_node.FunctionNode):

    @property
    def label(self):
        return 'digamma'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _digamma_cpu
        if _digamma_cpu is None:
            try:
                from scipy import special
                _digamma_cpu = special.digamma
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation'
                                  ' of digamma can not be done.')
        self.retain_inputs((0,))
        return utils.force_array(_digamma_cpu(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return utils.force_array(
            cuda.cupyx.scipy.special.digamma(x[0]), dtype=x[0].dtype),

    def backward(self, indexes, gy):
        z = self.get_retained_inputs()[0]
        xp = backend.get_array_module(*gy)
        return chainer.functions.polygamma(xp.array(1), z) * gy[0],


def digamma(x):
    """Digamma function.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return DiGamma().apply((x,))[0]
