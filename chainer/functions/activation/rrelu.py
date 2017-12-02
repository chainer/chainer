import chainer
from chainer import cuda
from chainer import function_node
from chainer.utils import type_check
import numpy as np


def _kern():
    return cuda.elementwise(
        'T cond, T x, T slope', 'T y',
        'y = cond >= 0 ? x : (T)(slope * x)', 'rrelu')


class RReLU(function_node.FunctionNode):
    """Randomized Leaky rectifier unit."""

    def __init__(self, lower=1. / 8, upper=1. / 3):
        # lower and upper must be [0, 1) and lower <= upper
        assert 0 <= lower < 1
        assert 0 <= upper < 1
        assert lower < upper
        self.lower = lower
        self.upper = upper

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        if hasattr(self, 'r'):
            y = np.where(x[0] >= 0, x[0], x[0]*self.r)
        else:
            if chainer.config.train:
                self.r = np.random.uniform(
                    self.lower, self.upper, x[0].shape).astype(x[0].dtype)
            else:
                self.r = np.empty(x[0].shape).astype(x[0].dtype)
                self.r.fill((self.lower + self.upper) / 2)
            y = np.where(x[0] >= 0, x[0], x[0]*self.r)
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, x):
        xp = cuda.cupy
        if hasattr(self, 'r'):
            y = _kern()(x[0], x[0], self.r.astype(x[0].dtype))
        else:
            if chainer.config.train:
                self.r = xp.random.uniform(
                    self.lower, self.upper, x[0].shape).astype(x[0].dtype)
            else:
                self.r = xp.empty(x[0].shape).astype(x[0].dtype)
                self.r.fill((self.lower + self.upper) / 2)
            y = _kern()(x[0], x[0], self.r)
        self.retain_inputs(())
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        x = None
        y = self.get_retained_outputs()[0].data
        return _RReLUGrad(x, y, self.r).apply(grad_outputs)


class _RReLUGrad(function_node.FunctionNode):
    def __init__(self, x, y, r):
        self.r = r
        self.x = x
        self.y = y

    def forward_cpu(self, inputs):
        gy, = inputs
        gy = gy.copy()
        gy = np.where(self.y >= 0, gy, gy*self.r)
        return gy,

    def forward_gpu(self, inputs):
        gy, = inputs
        gy = gy.copy()
        gy = _kern()(self.y, gy, self.r)
        return gy,

    def backward(self, indexes, grad_outputs):
        return _RReLUGrad(self.x, self.y, self.r).apply(grad_outputs)


def rrelu(x, l=1. / 8, u=1. / 3):
    """Randomized Leaky Rectified Liner Unit function.

    This function is expressed as

    .. math:: f(x)=\\max(x, ax),

    where :math:`a` is a random number sampled \
                from a uniform distribution :math:`U(l, u)`.

    See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        l (float): The lower bound of the uniform distribution.
        u (float): The upper bound of the uniform distribution.

    Returns:
        ~chainer.Variable: Outputs variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> x
        array([[-1.,  0.],
               [ 2., -3.],
               [-2.,  1.]], dtype=float32)
        >>> randomized_leaky_relu(x).data
        array([[-0.24850948,  0.        ],
               [ 2.        , -0.50844127],
               [-0.598535  ,  1.        ]], dtype=float32)
    """
    return RReLU(l, u).apply((x,))[0]
