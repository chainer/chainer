import chainer
from chainer import function_node
from chainer.utils import type_check
import chainerx


class Cholesky(function_node.FunctionNode):

    @property
    def label(self):
        return 'cholesky'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a', ))
        a_type, = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            a_type.ndim == 2,
        )

    def forward(self, inputs):
        a, = inputs
        self.retain_outputs((0,))
        xp = chainer.backend.get_array_module(a)
        return xp.linalg.cholesky(a),

    def forward_chainerx(self, inputs):
        return chainerx.linalg.cholesky(*inputs),

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
        gx = 0.5 * (s + s.T)
        return gx,


def cholesky(a):
    """Cholesky Decomposition

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cholesky().apply((a,))[0]
