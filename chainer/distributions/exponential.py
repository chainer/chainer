import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import where
from chainer.functions.math import exponential
from chainer.functions.math import exponential_m1
from chainer.functions.math import logarithm_1p
from chainer.utils import cache


class Exponential(distribution.Distribution):

    """Exponential Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\lambda) = \\lambda e^{-\\lambda x}

    Args:
        lam(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`\\lambda`.
    """

    def __init__(self, lam):
        super(Exponential, self).__init__()
        self.__lam = lam

    @cache.cached_property
    def lam(self):
        return chainer.as_variable(self.__lam)

    @cache.cached_property
    def _log_lam(self):
        return exponential.log(self.lam)

    @property
    def batch_shape(self):
        return self.lam.shape

    def cdf(self, x):
        return - exponential_m1.expm1(-self.lam * x)

    @cache.cached_property
    def entropy(self):
        return 1 - self._log_lam

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        x = chainer.as_variable(x)
        return -1 / self.lam * logarithm_1p.log1p(-x)

    @property
    def _is_gpu(self):
        return isinstance(self.lam.data, cuda.ndarray)

    def log_prob(self, x):
        logp = self._log_lam - self.lam * x
        xp = logp.xp
        if isinstance(x, chainer.Variable):
            x = x.array
        inf = xp.full_like(logp.array, xp.inf)
        return where.where(xp.asarray(x >= 0), logp, xp.asarray(-inf))

    @cache.cached_property
    def mean(self):
        return 1 / self.lam

    @property
    def params(self):
        return {'lam': self.lam}

    def sample_n(self, n):
        xp = chainer.backend.get_array_module(self.lam)
        if xp is cuda.cupy:
            eps = xp.random.standard_exponential(
                (n,)+self.lam.shape, dtype=self.lam.dtype)
        else:
            eps = xp.random.standard_exponential(
                (n,)+self.lam.shape).astype(self.lam.dtype)
        noise = eps / self.lam
        return noise

    @property
    def support(self):
        return 'positive'

    @cache.cached_property
    def variance(self):
        return self.mean ** 2


@distribution.register_kl(Exponential, Exponential)
def _kl_exponential_exponential(dist1, dist2):
    return (
        dist1._log_lam
        - dist2._log_lam
        + dist2.lam / dist1.lam
        - 1.)
