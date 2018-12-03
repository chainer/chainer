import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma


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
        self.__loc = chainer.as_variable(loc)
        self.__scale = chainer.as_variable(scale)

    @property
    def loc(self):
        return self.__loc

    @property
    def scale(self):
        return self.__scale

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def entropy(self):
        return exponential.log(self.scale) + (EULER + 1)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        y = (x - self.loc) / self.scale
        return - exponential.log(self.scale) - y - exponential.exp(-y)

    @property
    def mean(self):
        return self.loc + EULER * self.scale

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

    @property
    def variance(self):
        return numpy.pi ** 2 * self.scale ** 2 / 6


@distribution.register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(dist1, dist2):
    scale_1d2 = dist1.scale / dist2.scale
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + EULER * (scale_1d2 - 1.) \
        + exponential.exp((dist2.loc - dist1.loc) / dist2.scale
                          + lgamma.lgamma(scale_1d2 + 1.)) \
        - 1 + (dist1.loc - dist2.loc) / dist2.scale
