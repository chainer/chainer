import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class ELU(function_node.FunctionNode):

    """Exponential Linear Unit."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        if self.alpha < 0:
            self.retain_inputs((0,))
        x, = inputs
        y = x.copy()
        negzero_indices = y <= 0
        y[negzero_indices] = self.alpha * numpy.expm1(y[negzero_indices])
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, inputs):
        if self.alpha < 0:
            self.retain_inputs((0,))
        x, = inputs
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x > 0 ? x : (T)(alpha * expm1(x))',
            'elu_fwd')(x, self.alpha)
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        y, = self.get_retained_outputs()
        if self.alpha < 0:
            cond, = self.get_retained_inputs()
        else:
            cond = y
        gy, = grad_outputs
        return ELUGrad(self.alpha, cond.array).apply((y,))[0] * gy,


class ELUGrad(function_node.FunctionNode):

    """Exponential Linear Unit gradient function."""

    def __init__(self, alpha, cond):
        self.alpha = alpha
        self.cond = cond

    def forward_cpu(self, inputs):
        y, = inputs
        gx = utils.force_array(y + y.dtype.type(self.alpha))
        gx[self.cond > 0] = 1
        return gx,

    def forward_gpu(self, inputs):
        y, = inputs
        gx = cuda.elementwise(
            'T y, T alpha, T cond', 'T gx',
            'gx = cond > 0 ? (T)1 : (T)(y + alpha)',
            'elu_bwd')(y, self.alpha, self.cond)
        return gx,

    def backward(self, indexes, grad_outputs):
        ggx, = grad_outputs
        gy2 = ggx * (self.cond <= 0)
        return gy2,


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
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
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
        >>> y.array
        array([[-0.63212055,  0.        ],
               [ 2.        , -0.95021296]], dtype=float32)

    """
    return ELU(alpha=alpha).apply((x,))[0]
