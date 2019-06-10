from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_polygamma_cpu = None


class PolyGamma(function_node.FunctionNode):

    @property
    def label(self):
        return 'polygamma'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('n', 'x'))
        n_type, x_type = in_types

        type_check.expect(
            n_type.dtype.kind == 'i',
            x_type.dtype.kind == 'f',
        )

    def forward_cpu(self, inputs):
        n, x = inputs
        global _polygamma_cpu
        if _polygamma_cpu is None:
            try:
                from scipy import special
                _polygamma_cpu = special.polygamma
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation'
                                  ' of polygamma can not be done.')
        self.retain_inputs((0, 1))
        return utils.force_array(_polygamma_cpu(n, x), dtype=x.dtype),

    def forward_gpu(self, inputs):
        n, x = inputs
        self.retain_inputs((0, 1))
        return utils.force_array(
            cuda.cupyx.scipy.special.polygamma(n, x), dtype=x.dtype),

    def backward(self, indexes, gy):
        n, x = self.get_retained_inputs()
        return None, polygamma(n + 1, x) * gy[0],


def polygamma(n, x):
    """Polygamma function.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        n (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return PolyGamma().apply((n, x))[0]
