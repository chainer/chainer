import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma


class Chisquare(distribution.Distribution):

    """Chi-Square Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;k) = \\frac{1}{2^{k/2}\\Gamma(k/2)}x^{k/2-1}e^{-x/2}

    Args:
        k(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
    """

    def __init__(self, k):
        super(Chisquare, self).__init__()
        self.__k = chainer.as_variable(k)

    @property
    def k(self):
        return self.__k

    @property
    def batch_shape(self):
        return self.k.shape

    @property
    def entropy(self):
        return 0.5 * self.k + numpy.log(2.) + lgamma.lgamma(0.5 * self.k) \
            + (1 - 0.5 * self.k) * digamma.digamma(0.5 * self.k)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, x):
        return - lgamma.lgamma(0.5 * self.k) - 0.5 * self.k * numpy.log(2.) \
            + (0.5 * self.k - 1) * exponential.log(x) - 0.5 * x

    @property
    def mean(self):
        return self.k

    def sample_n(self, n):
        xp = cuda.get_array_module(self.k)
        if xp is cuda.cupy:
            eps = xp.random.chisquare(
                self.k.data, (n,)+self.k.shape, dtype=self.k.dtype)
        else:
            eps = xp.random.chisquare(
                self.k.data, (n,)+self.k.shape).astype(self.k.dtype)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        return 'positive'

    @property
    def variance(self):
        return 2 * self.k
