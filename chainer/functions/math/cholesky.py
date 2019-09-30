from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


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
                from scipy import linalg
                _cholesky_cpu = linalg.cholesky
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation'
                                  ' of cholesky can not be done.')
        self.retain_inputs((0, ))
        return _cholesky_cpu(a),

    def forward_gpu(self, inputs):
        a, = inputs
        self.retain_inputs((0, ))
        return cuda.cupy.linalg.cholesky(cuda.cupy.array(a.array)),

    def backward(self, indexes, gy):
        a, = self.get_retained_inputs().array
        return cuda.cupy.dot(a, a.T) * gy[0],


def cholesky(a):
    """Cholesky Decomposition function.

    .. note::
       Forward computation in CPU can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cholesky().apply((a,))[0]
