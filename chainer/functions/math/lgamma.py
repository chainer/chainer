import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

_lgamma_cpu = None


class LGamma(function_node.FunctionNode):

    @property
    def label(self):
        return 'lgamma'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _lgamma_cpu
        if _lgamma_cpu is None:
            try:
                from scipy import special
                _lgamma_cpu = special.gammaln
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation'
                                  ' of lgamma can not be done.')
        self.retain_inputs((0,))
        return utils.force_array(_lgamma_cpu(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return utils.force_array(
            cuda.cupyx.scipy.special.gammaln(x[0]), dtype=x[0].dtype),

    def backward(self, indexes, gy):
        z = self.get_retained_inputs()[0]
        return chainer.functions.digamma(z) * gy[0],


def lgamma(x):
    """logarithm of gamma function.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return LGamma().apply((x,))[0]
