import chainer
from chainer import backend
from chainer import distribution
from chainer.functions.array import where
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer import utils
from chainer.utils import cache


def _lbeta(a, b):
    return lgamma.lgamma(a) + lgamma.lgamma(b) - lgamma.lgamma(a + b)


class Beta(distribution.Distribution):

    """Beta Distribution.

    The probability density function of the distribution is expressed as

    .. math::
       f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)},

    for :math:`0 < x < 1`, :math:`\\alpha > 0`, :math:`\\beta > 0`.

    Args:
        a(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing :math:`\\alpha`.
        b(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing :math:`\\beta`.

    """

    def __init__(self, a, b):
        super(Beta, self).__init__()
        self.__a = a
        self.__b = b

    @cache.cached_property
    def a(self):
        return chainer.as_variable(self.__a)

    @cache.cached_property
    def b(self):
        return chainer.as_variable(self.__b)

    @cache.cached_property
    def _a_plus_b(self):
        return self.a + self.b

    @property
    def batch_shape(self):
        return self.a.shape

    @cache.cached_property
    def entropy(self):
        apb = self._a_plus_b
        return _lbeta(self.a, self.b) \
            - (self.a - 1) * digamma.digamma(self.a) \
            - (self.b - 1) * digamma.digamma(self.b) \
            + (apb - 2) * digamma.digamma(apb)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, x):
        x = chainer.as_variable(x)
        logp = (self.a - 1) * exponential.log(x) \
            + (self.b - 1) * exponential.log(1 - x) \
            - _lbeta(self.a, self.b)
        xp = logp.xp
        return where.where(
            utils.force_array((x.array >= 0) & (x.array <= 1)),
            logp, xp.array(-xp.inf, logp.dtype))

    @cache.cached_property
    def mean(self):
        return self.a / self._a_plus_b

    @property
    def params(self):
        return {'a': self.a, 'b': self.b}

    def sample_n(self, n):
        xp = backend.get_array_module(self.a)
        eps = xp.random.beta(self.a.data, self.b.data, size=(n,)+self.a.shape)
        noise = chainer.Variable(eps.astype(self.a.dtype))
        return noise

    @property
    def support(self):
        return '[0, 1]'

    @cache.cached_property
    def variance(self):
        apb = self._a_plus_b
        return self.a * self.b / apb ** 2 / (apb + 1)


@distribution.register_kl(Beta, Beta)
def _kl_beta_beta(dist1, dist2):
    dist1_apb = dist1._a_plus_b
    dist2_apb = dist2._a_plus_b
    return - _lbeta(dist1.a, dist1.b) + _lbeta(dist2.a, dist2.b)\
        + (dist1.a - dist2.a) * digamma.digamma(dist1.a) \
        + (dist1.b - dist2.b) * digamma.digamma(dist1.b) \
        + (dist2_apb - dist1_apb) * digamma.digamma(dist1_apb)
