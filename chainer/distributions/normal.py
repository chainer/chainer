import math

import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import exponential
from chainer.functions.math import log_ndtr
from chainer.functions.math import ndtr
from chainer.functions.math import ndtri
from chainer.utils import argument


ENTROPYC = 0.5 * math.log(2 * math.pi * math.e)
LOGPROBC = - 0.5 * math.log(2 * math.pi)
PROBC = 1. / (2 * math.pi) ** 0.5


class Normal(distribution.Distribution):

    """Normal Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,\\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}
            \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`. This is the mean parameter.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\sigma`. Either `scale` or `log_scale` (not both) must \
        have a value.
        log_scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\log(\\sigma)`. Either `scale` or `log_scale` (not \
        both) must have a value.

    """

    def __init__(self, loc, scale=None, **kwargs):
        super(Normal, self).__init__()
        log_scale = None
        if kwargs:
            log_scale, = argument.parse_kwargs(
                kwargs, ('log_scale', log_scale))
        if not (scale is None) ^ (log_scale is None):
            raise ValueError(
                "Either `scale` or `log_scale` (not both) must have a value.")
        self.loc = chainer.as_variable(loc)

        with chainer.using_config('enable_backprop', True):
            if scale is None:
                self.__log_scale = chainer.as_variable(log_scale)
                self.__scale = exponential.exp(self.log_scale)
            else:
                self.__scale = chainer.as_variable(scale)
                self.__log_scale = exponential.log(self.scale)

    @property
    def scale(self):
        return self.__scale

    @property
    def log_scale(self):
        return self.__log_scale

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        return ndtr.ndtr((x - self.loc) / self.scale)

    @property
    def entropy(self):
        return self.log_scale + ENTROPYC

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc + self.scale * ndtri.ndtri(x)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_cdf(self, x):
        return log_ndtr.log_ndtr((x - self.loc) / self.scale)

    def log_prob(self, x):
        return LOGPROBC - self.log_scale \
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2

    def log_survival_function(self, x):
        return log_ndtr.log_ndtr((self.loc - x) / self.scale)

    @property
    def mean(self):
        return self.loc

    @property
    def params(self):
        return {'loc': self.loc, 'scale': self.scale}

    def prob(self, x):
        return (PROBC / self.scale) * exponential.exp(
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2)

    def sample_n(self, n):
        if self._is_gpu:
            eps = cuda.cupy.random.standard_normal(
                (n,)+self.loc.shape, dtype=self.loc.dtype)
        else:
            eps = numpy.random.standard_normal(
                (n,)+self.loc.shape).astype(numpy.float32)
        noise = repeat.repeat(
            expand_dims.expand_dims(self.scale, axis=0), n, axis=0) * eps
        noise += repeat.repeat(expand_dims.expand_dims(
            self.loc, axis=0), n, axis=0)

        return noise

    @property
    def stddev(self):
        return self.scale

    @property
    def support(self):
        return 'real'

    def survival_function(self, x):
        return ndtr.ndtr((self.loc - x) / self.scale)

    @property
    def variance(self):
        return self.scale ** 2


@distribution.register_kl(Normal, Normal)
def _kl_normal_normal(dist1, dist2):
    return dist2.log_scale - dist1.log_scale \
        + 0.5 * (dist1.scale ** 2 + (dist1.loc - dist2.loc) ** 2) \
        / dist2.scale ** 2 - 0.5
