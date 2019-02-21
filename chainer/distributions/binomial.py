import numpy
import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.utils import cache


class Binomial(distribution.Distribution):

    """Binomial Distribution.

    The probability mass function of the distribution is expressed as:

    .. math::
        Pr(x = k) = (n k) p^{k} (1-p)^{n-k}
        for k = 1, 2, 3, ...,

    Args:
        p(:class:`~chainer.Variable` or :ref:`ndarray`):
            Success probability for each trial.
        n(:class:`~chainer.Variable` or :ref:`ndarray`):
            Number of total trials in distribution.
    """

    def __init__(self, n, p):
        super(Binomial, self).__init__()
        self.__n = n
        self.__p = p

    @cache.cached_property
    def n(self):
        return chainer.as_variable(self.__n)

    @cache.cached_property
    def p(self):
        return chainer.as_variable(self.__p)

    @property
    def batch_shape(self):
        return self.p.shape

    @property
    def event_shape(self):
        return ()

    @cache.cached_property
    def mean(self):
        return self.n * self.p

    @cache.cached_property
    def variance(self):
        return self.n * self.p * (1 - self.p)

    def sample_n(self, k):
        xp = cuda.get_array_module(self.p)
        if xp is cuda.cupy:
            eps = xp.random.binomial(
                self.n.data.astype('int64'),
                self.p.data,
                size=(k,)+self.batch_shape, dtype=self.p.dtype)
        else:
            eps = xp.random.binomial( 
                self.n.data.astype('int64'),
                self.p.data,
                size=(k,)+self.batch_shape).astype(self.p.dtype)
        return chainer.Variable(eps)

    @property
    def support(self):
        return 'positive integer'

@distribution.register_kl(Binomial, Binomial)
def _kl_binomial_binomial(dist1, dist2):
    return ((exponential.log(dist1.p) - exponential.log(dist2.p))
            * dist1.n * dist1.p + (exponential.log(1 - dist1.p)
            - exponential(1 - dist2.p)) * dist1.n * (1 - dist1.p)
            )
