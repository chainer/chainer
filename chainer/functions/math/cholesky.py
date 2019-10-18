import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
from chainer import utils

_cholesky_cpu = None


class Cholesky(function_node.FunctionNode):

    @property
    def label(self):
        return 'cholesky'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a'))
        a_type, = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            a_type.ndim == 2,
        )

    def forward_cpu(self, inputs):
        a, = inputs
        global _cholesky_cpu
        if _cholesky_cpu is None:
            from numpy import linalg
            _cholesky_cpu = linalg.cholesky
        self.retain_outputs((0,))
        return utils.force_array(_cholesky_cpu(a), dtype=a.dtype),

    def forward_gpu(self, inputs):
        a, = inputs
        self.retain_outputs((0,))
        return utils.force_array(
            cuda.cupy.linalg.cholesky(a), dtype=a.dtype),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        xp = chainer.backend.get_array_module(gy)
        y, = self.get_retained_outputs()
        n = y.shape[0]
        dtype = y.dtype

        F = chainer.functions
        y_inv = F.inv(y)
        mask = xp.tri(n, dtype=dtype) - 0.5 * xp.eye(n, dtype=dtype)
        phi = mask * F.matmul(y, gy, transa=True)
        s = F.matmul(F.matmul(y_inv, phi, transa=True), y_inv)
        gx = mask * (s + s.T)
        return gx,


def cholesky(a):
    """Cholesky Decomposition

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cholesky().apply((a,))[0]
