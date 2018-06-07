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
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, z):
        global _lgamma_cpu
        if _lgamma_cpu is None:
            try:
                from scipy import special
                _lgamma_cpu = special.gammaln
            except ImportError:
                raise ImportError("SciPy is not available. Forward computation"
                                  " of lgamma can not be done.")
        self.retain_inputs((0,))
        return utils.force_array(_lgamma_cpu(z[0]), dtype=z[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            'y = lgammaf(x)',
            'elementwise_lgammaf',
        )(x[0]),

    def backward(self, indexes, gy):
        z = self.get_retained_inputs()[0]
        return chainer.functions.digamma(z) * gy[0],


def lgamma(z):
    """logarithm of gamma function.

    .. note::
       Forward computation can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        z (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return LGamma().apply((z,))[0]
