from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy as np


def _kern():
    return cuda.elementwise(
        'T cond, T x, T slope', 'T y',
        'y = cond >= 0 ? x : (T)(slope * x)', 'rrelu')


class RReLU(function.Function):
    """Randomized Leaky rectifier unit."""

    def __init__(self, lower=1./8, upper=1./3, train=True):
        self.lower = lower
        self.upper = upper
        self.train = train

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        y = x[0].copy()
        if self.train:
            self.r = np.random.uniform(self.lower, self.upper, x[0].shape[0:2])
        else:
            self.r = np.empty(x[0].shape[0:2])
            self.r.fill((self.lower+self.upper)/2)
        y *= np.where(x[0] < 0, self.r, 1)
        if self.lower >= 0 and self.upper >= 0:
            self.retain_inputs(())
            self.retain_outputs((0,))
        return y,

    def forward_gpu(self, x):
        xp = cuda.cupy
        if self.train:
            self.r = xp.random.uniform(
                self.lower, self.upper, x[0].shape[:2]
            ).astype(x[0].dtype)
        else:
            self.r = xp.empty(x[0].shape[:2])
            self.r.fill((self.lower+self.upper)/2.0)
            self.r = self.r.astype(x[0].dtype)
        y = _kern()(x[0], x[0], self.r)
        if self.lower >= 0 and self.upper >= 0:
            self.retain_inputs(())
            self.retain_outputs((0,))
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        if self.lower >= 0 and self.upper >= 0:
            y = self.output_data
            gx *= np.where(y[0] < 0, self.r, 1)
        else:
            gx *= np.where(x[0] < 0, self.r, 1)
        return gx,

    def backward_gpu(self, x, gy):
        if self.lower >= 0 and self.upper >= 0:
            y = self.output_data
            gx = _kern()(y[0], gy[0], self.r)
        else:
            gx = _kern()(x[0], gy[0], self.r)
        return gx,


def randomized_leaky_relu(x, l=1./8, u=1./3, train=True):
    """Randomized Leaky Rectified Liner Unit function.

    This function is expressed as

    .. math:: f(x)=\\max(x, ax),

    where :math:`a` is a random number sampled \
                from a uniform distribution U(l, u).

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        l (float):
        r (float):

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
    return RReLU(l, u, train)(x)
