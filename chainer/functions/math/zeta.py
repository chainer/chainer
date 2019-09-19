from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_zeta_cpu = None


class Zeta(function_node.FunctionNode):

    def __init__(self, x):
        self._x = x

    @property
    def label(self):
        return 'zeta'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('q'))
        q_type, = in_types
        type_check.expect(
            q_type.dtype.kind == 'f'
        )

    def forward_cpu(self, inputs):
        q, = inputs
        global _zeta_cpu
        if _zeta_cpu is None:
            try:
                from scipy import special
                _zeta_cpu = special.zeta
            except ImportError:
                raise ImportError('Scipy is not available. Forward computation'
                                  ' of zeta cannot be done.')
        self.retain_inputs((0,))
        return utils.force_array(_zeta_cpu(self._x, q), dtype=q.dtype),

    def forward_gpu(self, inputs):
        q, = inputs
        self.retain_inputs((0,))
        return utils.force_array(
            cuda.cupyx.scipy.special.zeta(self._x, q), dtype=q.dtype),

    def backward(self, indexes, gy):
        q, = self.get_retained_inputs()
        return gy[0] * -self._x * zeta(self._x + 1, q),


def zeta(x, q):
    """Zeta function.

    Differentiable only with respect to q

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        q (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Zeta(x).apply((q,))[0]
