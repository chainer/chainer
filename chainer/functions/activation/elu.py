import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class ELU(function_node.FunctionNode):

    """Exponential Linear Unit."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.expm1(y[neg_indices]))
        return y,

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : (T)(alpha * expm1(x))',
            'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        return ELUGrad(self.alpha).apply((x,))[0] * gy,


class ELUGrad(function_node.FunctionNode):

    """Exponential Linear Unit gradient function."""

    def __init__(self, alpha):
        self.alpha = alpha

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x, = inputs
        gx = numpy.ones_like(x)
        neg_indices = x < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[neg_indices])
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return gx,

    def forward_gpu(self, inputs):
        x, = inputs
        gx = cuda.elementwise(
            'T x, T alpha', 'T gx',
            'gx = x >= 0 ? (T)1 : (T)(alpha * exp(x))',
            'elu_bwd')(
                x, self.alpha)
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return gx,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gx, = self.get_retained_outputs()
        ggx, = grad_outputs
        return ggx * gx * (x.data < 0),


def elu(x, alpha=1.0):
    """Exponential Linear Unit function.

    For a parameter :math:`\\alpha`, it is expressed as

    .. math::
        f(x) = \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.

    See: https://arxiv.org/abs/1511.07289

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        alpha (float): Parameter :math:`\\alpha`. Default is 1.0.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3]], np.float32)
        >>> x
        array([[-1.,  0.],
               [ 2., -3.]], dtype=float32)
        >>> y = F.elu(x, alpha=1.)
        >>> y.data
        array([[-0.63212055,  0.        ],
               [ 2.        , -0.95021296]], dtype=float32)

    """
    return ELU(alpha=alpha).apply((x,))[0]
