import math

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import diagonal
from chainer.functions.array import expand_dims
from chainer.functions.array import squeeze
from chainer.functions.array import stack
from chainer.functions.array import swapaxes
from chainer.functions.array import transpose
from chainer.functions.array import where
from chainer.functions.math import exponential
from chainer.functions.math import matmul
from chainer.functions.math import sum as sum_mod
from chainer.utils import argument
from chainer.utils import cache
from chainer.utils import type_check

try:
    import scipy.linalg
    available_cpu = True
except ImportError as e:
    available_cpu = False
    _import_error = e

ENTROPYC = 0.5 * math.log(2 * math.pi * math.e)
LOGPROBC = - 0.5 * math.log(2 * math.pi)


class TriangularInv(chainer.function_node.FunctionNode):

    def __init__(self, lower):
        self._lower = lower

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types
        type_check.expect(a_type.dtype == numpy.float32)
        # Only 2D array shapes allowed
        type_check.expect(a_type.ndim == 2)
        # Matrix inversion only allowed for square matrices
        type_check.expect(a_type.shape[0] == a_type.shape[1])

    def forward_cpu(self, inputs):
        self.retain_outputs((0,))
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation'
                              ' of triangular_inv in CPU can not be done.' +
                              str(_import_error))
        x, = inputs
        if len(x) == 0:
            # linalg.solve_triangular crashes
            return x,
        invx = scipy.linalg.solve_triangular(
            x, numpy.eye(len(x), dtype=x.dtype), lower=self._lower)
        return invx,

    def forward_gpu(self, inputs):
        self.retain_outputs((0,))
        x, = inputs
        if len(x) == 0:
            # linalg.solve_triangular crashes
            return x,
        invx = cuda.cupyx.scipy.linalg.solve_triangular(
            x, cuda.cupy.eye(len(x), dtype=x.dtype), lower=self._lower)
        return invx,

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        xp = backend.get_array_module(gy)
        invx, = self.get_retained_outputs()
        mask = xp.tril(xp.ones((len(invx), len(invx)), dtype=bool))
        if not self._lower:
            mask = mask.T
        # Gradient is - x^-T (dx) x^-T
        invxT = chainer.functions.transpose(invx)
        gx = chainer.functions.matmul(
            chainer.functions.matmul(- invxT, gy), invxT)
        gx = where.where(mask, gx, xp.zeros_like(gx.array))
        return gx,


def _triangular_inv(x, lower=True):
    y, = TriangularInv(lower).apply((x,))
    return y


def _batch_triangular_inv(x, lower=True):
    n = len(x)
    y = []
    for i in range(n):
        y.append(_triangular_inv(x[i]))
    return stack.stack(y)


def _triangular_logdet(x):
    diag = diagonal.diagonal(x, axis1=-2, axis2=-1)
    return sum_mod.sum(exponential.log(abs(diag)), axis=-1)


class MultivariateNormal(distribution.Distribution):

    """MultivariateNormal Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,V) = \\frac{1}{\\sqrt{\\det(2\\pi V)}}
            \\exp\\left(-\\frac{1}{2}(x-\\mu) V^{-1}(x-\\mu)\\right)

    Args:
        loc (:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the location :math:`\\mu`.
        scale_tril (:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the scale :math:`L` such that
            :math:`V=LL^T`.

    """

    def __init__(self, loc, **kwargs):
        scale_tril = None
        if kwargs:
            scale_tril, = argument.parse_kwargs(
                kwargs, ('scale_tril', scale_tril))
        if scale_tril is None:
            raise ValueError('`scale_tril` must have a value.')
        self.__loc = loc
        self.__scale_tril = scale_tril

    @cache.cached_property
    def loc(self):
        return chainer.as_variable(self.__loc)

    @cache.cached_property
    def scale_tril(self):
        return chainer.as_variable(self.__scale_tril)

    @cache.cached_property
    def _logdet_scale(self):
        return _triangular_logdet(self.scale_tril)

    @property
    def d(self):
        return self.scale_tril.shape[-1]

    def __copy__(self):
        return self._copy_to(MultivariateNormal(self.loc, self.scale_tril))

    @property
    def batch_shape(self):
        return self.loc.shape[:-1]

    @cache.cached_property
    def entropy(self):
        return self._logdet_scale + ENTROPYC * self.d

    @property
    def event_shape(self):
        return self.loc.shape[-1:]

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        scale_tril_inv = \
            _batch_triangular_inv(self.scale_tril.reshape(-1, self.d, self.d))
        scale_tril_inv = scale_tril_inv.reshape(
            self.batch_shape+(self.d, self.d))

        bsti = broadcast.broadcast_to(scale_tril_inv, x.shape + (self.d,))
        bl = broadcast.broadcast_to(self.loc, x.shape)
        m = matmul.matmul(
            bsti,
            expand_dims.expand_dims(x - bl, axis=-1))
        m = matmul.matmul(swapaxes.swapaxes(m, -1, -2), m)
        m = squeeze.squeeze(m, axis=-1)
        m = squeeze.squeeze(m, axis=-1)
        logz = LOGPROBC * self.d - self._logdet_scale
        return broadcast.broadcast_to(logz, m.shape) - 0.5 * m

    @cache.cached_property
    def mean(self):
        return self.loc

    def sample_n(self, n):
        if self._is_gpu:
            eps = cuda.cupy.random.standard_normal(
                (n,)+self.loc.shape+(1,), dtype=self.loc.dtype)
        else:
            eps = numpy.random.standard_normal(
                (n,)+self.loc.shape+(1,)).astype(numpy.float32)

        return self.loc + squeeze.squeeze(
            matmul.matmul(self.scale_tril, eps), axis=-1)

    @property
    def support(self):
        return 'real'

    @property
    def params(self):
        return {'loc': self.loc, 'scale_tril': self.scale_tril}

    @property
    def covariance(self):
        return matmul.matmul(
            self.scale_tril, transpose.transpose(
                self.scale_tril,
                tuple(range(len(self.batch_shape))) + (-1, -2)))


@distribution.register_kl(MultivariateNormal, MultivariateNormal)
def _kl_multivariatenormal_multivariatenormal(dist1, dist2):
    scale_tril_inv2 = _batch_triangular_inv(dist2.scale_tril.reshape(
        -1, dist2.d, dist2.d))
    trace = sum_mod.sum(matmul.matmul(
        scale_tril_inv2, dist1.scale_tril.reshape(-1, dist2.d, dist2.d)) ** 2,
        axis=(-1, -2)).reshape(dist1.batch_shape)

    mu = dist1.loc - dist2.loc
    mah = matmul.matmul(scale_tril_inv2, mu.reshape(-1, dist1.d, 1))
    mah = sum_mod.sum(mah ** 2, axis=-2).reshape(dist1.batch_shape)
    return dist2._logdet_scale - dist1._logdet_scale \
        + 0.5 * trace + 0.5 * mah - 0.5 * dist1.d
