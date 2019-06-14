import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.utils import cache


EULER = 0.57721566490153286060651209008240243104215933593992


class Gumbel(distribution.Distribution):

    """Gumbel Distribution.

    The probability density function of the distribution is expressed as

    .. math::
       f(x) = \\frac{1}{\\eta} \
           \\exp\\left\\{ - \\frac{x - \\mu}{\\eta} \\right\\} \
           \\exp\\left[-\\exp\\left\\{-\\frac{x - \\mu}{\\eta} \
           \\right\\}\\right],

    Args:
        loc(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`\\eta`.
    """

    def __init__(self, loc, scale):
        super(Gumbel, self).__init__()
        self.__loc = loc
        self.__scale = scale

    @cache.cached_property
    def loc(self):
        return chainer.as_variable(self.__loc)

    @cache.cached_property
    def scale(self):
        return chainer.as_variable(self.__scale)

    @cache.cached_property
    def _log_scale(self):
        return exponential.log(self.scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    @cache.cached_property
    def entropy(self):
        return self._log_scale + (EULER + 1)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        y = (x - self.loc) / self.scale
        return - self._log_scale - y - exponential.exp(-y)

    @cache.cached_property
    def mean(self):
        return self.loc + EULER * self.scale

    @property
    def params(self):
        return {'loc': self.loc, 'scale': self.scale}

    def sample_n(self, n):
        xp = cuda.get_array_module(self.loc)
        if xp is cuda.cupy:
            eps = xp.random.gumbel(
                size=(n,)+self.batch_shape, dtype=self.loc.dtype)
        else:
            eps = xp.random.gumbel(
                size=(n,)+self.batch_shape).astype(self.loc.dtype)
        noise = self.scale * eps + self.loc
        return noise

    @property
    def support(self):
        return 'real'

    @cache.cached_property
    def variance(self):
        return (numpy.pi ** 2 / 6) * self.scale ** 2


@distribution.register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(dist1, dist2):
    scale_1d2 = dist1.scale / dist2.scale
    return dist2._log_scale - dist1._log_scale \
        + EULER * (scale_1d2 - 1.) \
        + exponential.exp((dist2.loc - dist1.loc) / dist2.scale
                          + lgamma.lgamma(scale_1d2 + 1.)) \
        - 1 + (dist1.loc - dist2.loc) / dist2.scale
