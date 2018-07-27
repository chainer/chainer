import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import where
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class Binomial(distribution.Distribution):

    """Binomial Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        P(x = k; n, p) = \\binom{n}{k}p^k(1-p)^{n-k}

    Args:
        n(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, n, p):
        self.__n = chainer.as_variable(n)
        self.__p = chainer.as_variable(p)

    @property
    def n(self):
        return self.__n

    @property
    def p(self):
        return self.__p

    @property
    def batch_shape(self):
        return self.n.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.n.data, cuda.ndarray)

    def log_prob(self, x):
        n = self.n.data.astype(self.p.dtype)
        if isinstance(x, chainer.Variable):
            x = x.data.astype(self.p.dtype)
        else:
            x = x.astype(self.p.dtype)
        xp = cuda.get_array_module(x)

        bp = broadcast.broadcast_to(self.p, x.shape)
        bn = xp.broadcast_to(n, x.shape)

        constraint = xp.bitwise_and(x >= 0, x <= bn)

        log_p = lgamma.lgamma(bn + 1) - lgamma.lgamma(x + 1) \
            - lgamma.lgamma(bn - x + 1) + x * exponential.log(bp) \
            + (bn - x) * exponential.log(1 - bp)
        return where.where(constraint, log_p, - numpy.inf * xp.ones_like(x))

    @property
    def mean(self):
        return self.n.data * self.p

    def sample_n(self, n):
        xp = cuda.get_array_module(self.p)
        eps = xp.random.binomial(self.n.data, self.p.data,
                                 size=(n,)+self.n.shape)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        return '[0, n]'

    @property
    def variance(self):
        return self.n.data * self.p * (1 - self.p)


@distribution.register_kl(Binomial, Binomial)
def _kl_binomial_binomial(dist1, dist2):
    if (dist1.n.data < dist2.n.data).any():
        raise NotImplementedError()
    n1 = dist1.n.data.astype(dist1.p.dtype)
    xp = cuda.get_array_module(dist1.p)
    is_inf = dist1.n.data > dist2.n.data

    kl = n1 * dist1.p * (exponential.log(dist1.p)
                         - exponential.log(dist2.p)) \
        + n1 * (1 - dist1.p) * (exponential.log(1 - dist1.p)
                                - exponential.log(1 - dist2.p))

    return where.where(is_inf, numpy.inf * xp.ones_like(kl.data), kl)
