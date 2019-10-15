from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
from chainer import utils
from chainer import functions

_cholesky_cpu = None


class Cholesky(function_node.FunctionNode):

    @property
    def label(self):
        return 'cholesky'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a'))
        a_type, = in_types

        type_check.expect(
            a_type.dtype.kind == 'f'
        )

    def forward_cpu(self, inputs):
        a, = inputs
        global _cholesky_cpu
        if _cholesky_cpu is None:
            try:
                from numpy import linalg
                _cholesky_cpu = linalg.cholesky
            except ImportError:
                raise ImportError('NumPy is not available. Forward computation'
                                  ' of cholesky can not be done.')
        self.retain_inputs((0, ))
        return utils.force_array(_cholesky_cpu(a), dtype=a.dtype),

    def forward_gpu(self, inputs):
        a, = inputs
        self.retain_inputs((0, ))
        return utils.force_array(
            cuda.cupy.linalg.cholesky(a), dtype=a.dtype),

    def backward(self, indexes, gy):
        a, = self.get_retained_inputs()
        return functions.matmul(a, a, transb=True) * gy[0],


def cholesky(a):
    """Cholesky Decomposition

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cholesky().apply((a,))[0]
