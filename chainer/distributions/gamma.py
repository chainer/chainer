import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import where
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.utils import cache


class Gamma(distribution.Distribution):

    """Gamma Distribution.

    Args:
        k(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
        theta(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
    """

    def __init__(self, k, theta):
        super(Gamma, self).__init__()
        self.__k = k
        self.__theta = theta

    @cache.cached_property
    def k(self):
        return chainer.as_variable(self.__k)

    @cache.cached_property
    def theta(self):
        return chainer.as_variable(self.__theta)

    @property
    def batch_shape(self):
        return self.k.shape

    @cache.cached_property
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
        logp = - lgamma.lgamma(self.k) - self.k * exponential.log(self.theta) \
            + (self.k - 1) * exponential.log(x) - x / self.theta
        xp = logp.xp
        inf = xp.full_like(logp.array, xp.inf)
        if isinstance(x, chainer.Variable):
            x = x.array
        return where.where(xp.asarray(x >= 0), logp, xp.asarray(-inf))

    @cache.cached_property
    def mean(self):
        return self.k * self.theta

    @property
    def params(self):
        return {'k': self.k, 'theta': self.theta}

    def sample_n(self, n):
        xp = cuda.get_array_module(self.k)
        if xp is cuda.cupy:
            eps = xp.random.gamma(
                self.k.data, size=(n,) + self.batch_shape, dtype=self.k.dtype)
        else:
            eps = xp.random.gamma(
                self.k.data, size=(n,) + self.batch_shape).astype(self.k.dtype)
        noise = broadcast.broadcast_to(self.theta, eps.shape) * eps
        return noise

    @property
    def support(self):
        return 'positive'

    @cache.cached_property
    def variance(self):
        return self.mean * self.theta


@distribution.register_kl(Gamma, Gamma)
def _kl_gamma_gamma(dist1, dist2):
    return (dist1.k - dist2.k) * digamma.digamma(dist1.k) \
        - (lgamma.lgamma(dist1.k) - lgamma.lgamma(dist2.k)) \
        + dist2.k\
        * (exponential.log(dist2.theta) - exponential.log(dist1.theta)) \
        + dist1.k * (dist1.theta / dist2.theta - 1)
