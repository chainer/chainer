import chainer
from chainer import distribution
from chainer.utils import cache

from scipy.stats import skellam


class Skellam(distribution.Distribution):

    """Skellam Distribution.
    See here for more details:
    https://en.wikipedia.org/wiki/Skellam_distribution

    The probability mass function of the distribution is expressed as

    .. math::
        Pr(x = k) = e^{-(mu1 + mu2)} (\\frac{mu1 mu2})^{k/2}
            I_k(2 sqrt(mu1 mu2))
        , where I_k is the modified Bessel function of the first kind

    Args:
        mu1(:class:`~chainer.Variable` or :ref:`ndarray`):
            Expected value for the first Poisson distribution.
        mu2(:class:`~chainer.Variable` or :ref:`ndarray`):
            Expected value for the second Poisson distribution.
    """

    def __init__(self, mu1, mu2):
        super(Skellam, self).__init__()
        self.__mu1 = mu1
        self.__mu2 = mu2

    @cache.cached_property
    def mu1(self):
        return chainer.as_variable(self.__mu1)

    @cache.cached_property
    def mu2(self):
        return chainer.as_variable(self.__mu2)

    @property
    def batch_shape(self):
        return self.mu1.shape

    @property
    def event_shape(self):
        return ()

    @cache.cached_property
    def mean(self):
        return self.mu1 - self.mu2

    @property
    def support(self):
        return 'positive integer'

    @cache.cached_property
    def variance(self):
        return self.mu1 + self.mu2

    def cdf(self, k):
        cdf = skellam.cdf(k.data, self.mu1.data, self.mu2.data)
        return chainer.as_variable(cdf)

    def log_cdf(self, k):
        log_cdf = skellam.logcdf(k.data, self.mu1.data, self.mu2.data)
        return chainer.as_variable(log_cdf)
