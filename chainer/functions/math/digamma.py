import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import numpy


_digamma_cpu = None


class DiGamma(function_node.FunctionNode):

    @property
    def label(self):
        return 'digamma'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, z):
        global _digamma_cpu
        if _digamma_cpu is None:
            try:
                from scipy import special
                _digamma_cpu = special.digamma
            except ImportError:
                raise ImportError("SciPy is not available. Forward computation"
                                  " of digamma can not be done.")
        self.retain_inputs((0,))
        return utils.force_array(_digamma_cpu(z[0]), dtype=z[0].dtype),

    def forward_gpu(self, z):
        global _digamma_cpu
        if _digamma_cpu is None:
            try:
                from scipy import special
                _digamma_cpu = special.digamma
            except ImportError:
                raise ImportError("SciPy is not available. Forward computation"
                                  " of digamma can not be done.")
        self.retain_inputs((0,))

        self._in_device = cuda.get_device_from_array(z[0]).id
        return utils.force_array(
            cuda.to_gpu(_digamma_cpu(cuda.to_cpu(z[0])),
                        self._in_device), dtype=z[0].dtype),

    def backward(self, indexes, gy):
        z = self.get_retained_inputs()[0]
        xp = cuda.get_array_module(*gy)
        return chainer.functions.polygamma(xp.array(1), z) * gy[0],


def digamma(z):
    """Digamma function.

    .. note::
       Forward computation can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        z (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return DiGamma().apply((z,))[0]
