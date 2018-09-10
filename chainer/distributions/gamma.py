import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import where
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma


class Gamma(distribution.Distribution):

    """Gamma Distribution.

    Args:
        k(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        theta(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, k, theta):
        super(Gamma, self).__init__()
        self.__k = chainer.as_variable(k)
        self.__theta = chainer.as_variable(theta)

    @property
    def k(self):
        return self.__k

    @property
    def theta(self):
        return self.__theta

    @property
    def batch_shape(self):
        return self.k.shape

    @property
    def entropy(self):
        return self.k + exponential.log(self.theta) + lgamma.lgamma(self.k) \
            + (1 - self.k) * digamma.digamma(self.k)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.k.data, cuda.ndarray)

    def log_prob(self, x):
        bk = broadcast.broadcast_to(self.k, x.shape)
        btheta = broadcast.broadcast_to(self.theta, x.shape)

        logp = - lgamma.lgamma(bk) - bk * exponential.log(btheta) \
            + (bk - 1) * exponential.log(x) - x / btheta
        return where.where(x.data >= 0, logp, -numpy.inf)

    @property
    def mean(self):
        return self.k * self.theta

    def sample_n(self, n):
        xp = cuda.get_array_module(self.low)
        if xp is cuda.cupy:
            eps = xp.random.gamma(
                self.k.data, size=(n,) + self.low.shape, dtype=self.low.dtype)
        else:
            eps = xp.random.gamma(
                self.k.data, size=(n,) + self.low.shape).astype(self.low.dtype)
        noise = broadcast.broadcast_to(self.theta, eps.shape) * eps
        return noise

    @property
    def support(self):
        return 'positive'

    @property
    def variance(self):
        return self.k * self.theta * self.theta


@distribution.register_kl(Gamma, Gamma)
def _kl_gamma_gamma(dist1, dist2):
    return (dist1.k - dist2.k) * digamma.digamma(dist1.k) \
        - (lgamma.lgamma(dist1.k) - lgamma.lgamma(dist2.k)) \
        + dist2.k\
        * (exponential.log(dist2.theta) - exponential.log(dist1.theta)) \
        + dist1.k * (dist1.theta / dist2.theta - 1)
