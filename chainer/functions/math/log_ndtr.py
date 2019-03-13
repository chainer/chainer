import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.functions.math import erfcx
from chainer import utils
from chainer.utils import type_check


_log_ndtr_cpu = None


class LogNdtr(function_node.FunctionNode):

    @property
    def label(self):
        return 'log_ndtr'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        global _log_ndtr_cpu
        if _log_ndtr_cpu is None:
            try:
                from scipy import special
                _log_ndtr_cpu = special.log_ndtr
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation'
                                  ' of log_ndtr can not be done.')

        self.retain_inputs((0,))
        return utils.force_array(_log_ndtr_cpu(x[0]), dtype=x[0].dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.elementwise(
            'T x', 'T y',
            '''
            if (x > 0) {
                y = log1p(-normcdf(-x));
            } else {
                y = log(0.5 * erfcx(-sqrt(0.5) * x)) - 0.5 * x * x;
            }
            ''',
            'elementwise_log_ndtr',
        )(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return (2 / numpy.pi) ** 0.5 / erfcx.erfcx(- x / 2 ** 0.5) * gy[0],


def log_ndtr(x):
    """Logarithm of cumulative distribution function of normal distribution.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return LogNdtr().apply((x,))[0]
