import math

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential

LOGPROBC = - 0.5 * math.log(2 * math.pi)


class LogNormal(distribution.Distribution):

    """Logatithm Normal Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,\\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}x}
            \\exp\\left(-\\frac{(\\log{x}-\\mu)^2}{2\\sigma^2}\\right)

    Args:
        mu(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`\\mu`.
        sigma(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`\\sigma`.
    """

    def __init__(self, mu, sigma):
        self.__mu = chainer.as_variable(mu)
        self.__sigma = chainer.as_variable(sigma)

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    @property
    def batch_shape(self):
        return self.mu.shape

    @property
    def entropy(self):
        return 0.5 - LOGPROBC + exponential.log(self.sigma) + self.mu

    @property
    def event_shape(self):
        return ()

    def log_prob(self, x):
        logx = exponential.log(x)
        return LOGPROBC - exponential.log(self.sigma) - logx \
            - (0.5 * (logx - self.mu) ** 2 / self.sigma ** 2)

    @property
    def mean(self):
        return exponential.exp(self.mu + 0.5 * self.sigma ** 2)

    def sample_n(self, n):
        xp = backend.get_array_module(self.mu)
        if xp is cuda.cupy:
            eps = xp.random.standard_normal(
                (n,)+self.mu.shape, dtype=self.mu.dtype)
        else:
            eps = xp.random.standard_normal(
                (n,)+self.mu.shape).astype(self.mu.dtype)

        noise = self.sigma * eps
        noise += self.mu

        return exponential.exp(noise)

    @property
    def support(self):
        return 'positive'

    @property
    def variance(self):
        return exponential.exp(2 * self.mu + self.sigma ** 2) \
            * (exponential.exp(self.sigma ** 2) - 1)


@distribution.register_kl(LogNormal, LogNormal)
def _kl_log_normal_log_normal(dist1, dist2):
    return 0.5 * ((dist1.mu - dist2.mu) ** 2 +
                  dist1.sigma ** 2) / dist2.sigma ** 2 - 0.5 \
        + exponential.log(dist2.sigma) - exponential.log(dist1.sigma)
