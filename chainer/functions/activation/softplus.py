import numpy

from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class Softplus(function_node.FunctionNode):

    """Softplus function."""

    def __init__(self, beta=1.0):
        self.beta = float(beta)
        self.beta_inv = float(1.0 / beta)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0]
        # y = log(1 + exp(beta * x)) / beta
        bx = self.beta * x
        y = (numpy.fmax(bx, 0) +
             numpy.log1p(numpy.exp(-numpy.fabs(bx)))) * self.beta_inv
        return utils.force_array(y, x.dtype),

    def forward_gpu(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0]
        y = cuda.elementwise(
            'T x, T beta, T beta_inv', 'T y',
            '''
              T bx = beta * x;
              y = (max(bx, (T)0) + log1p(exp(-fabs(bx)))) * beta_inv;
            ''',
            'softplus_fwd'
        )(x, self.beta, self.beta_inv)
        return y,

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0]
        gy, = grad_outputs
        return SoftplusGrad((self.beta,)).apply((x, gy))


class SoftplusGrad(function_node.FunctionNode):

    """Softplus gradient function."""

    def __init__(self, inputs):
        super(SoftplusGrad, self).__init__()
        self.beta = inputs[0]

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, gy_type = in_types
        type_check.expect(x_type.dtype.kind == 'f')
        type_check.expect(gy_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = (1 - 1 / (1 + numpy.exp(self.beta * x))) * gy
        return utils.force_array(gx, x.dtype),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy, T beta', 'T gx',
            'gx = (1 - 1 / (1 + exp(beta * x))) * gy',
            'softplus_bwd')(x, gy, self.beta)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggx, = grad_outputs
        e = chainer.functions.exp(self.beta * x)
        gx = ggx * gy * self.beta * e / (1 + e) ** 2
        ggy = SoftplusGrad((self.beta,)).apply((x, ggx))[0]
        return gx, ggy


def softplus(x, beta=1.0):
    """Element-wise softplus function.

    The softplus function is the smooth approximation of ReLU.

    .. math:: f(x)=\\frac{1}{\\beta}\\log(1 + \\exp(\\beta x)),

    where :math:`\\beta` is a parameter. The function becomes curved
    and akin to ReLU as the :math:`\\beta` is increasing.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        beta (float): Parameter :math:`\\beta`.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.arange(-2, 3, 2).astype(np.float32)
        >>> x
        array([-2.,  0.,  2.], dtype=float32)
        >>> F.softplus(x, beta=1.0).array
        array([0.126928 , 0.6931472, 2.126928 ], dtype=float32)

    """
    y, = Softplus(beta=beta).apply((x,))
    return y
