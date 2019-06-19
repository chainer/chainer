import math

import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import log_ndtr
from chainer.functions.math import ndtr
from chainer.functions.math import ndtri
from chainer.utils import argument
from chainer.utils import cache
import chainerx


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
        loc(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the location :math:`\\mu`. This is the
            mean parameter.
        scale(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the scale :math:`\\sigma`. Either `scale`
            or `log_scale` (not both) must have a value.
        log_scale(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the scale :math:`\\log(\\sigma)`. Either
            `scale` or `log_scale` (not both) must have a value.

    """

    def __init__(self, loc, scale=None, **kwargs):
        super(Normal, self).__init__()
        log_scale = None
        if kwargs:
            log_scale, = argument.parse_kwargs(
                kwargs, ('log_scale', log_scale))
        if not (scale is None) ^ (log_scale is None):
            raise ValueError(
                'Either `scale` or `log_scale` (not both) must have a value.')

        self.__loc = loc
        self.__scale = scale
        self.__log_scale = log_scale
        if isinstance(loc, chainer.Variable):
            self.__device = loc.device
        else:
            self.__device = chainer.backend.get_device_from_array(loc)

    @cache.cached_property
    def loc(self):
        return chainer.as_variable(self.__loc)

    @cache.cached_property
    def scale(self):
        if self.__scale is not None:
            return chainer.as_variable(self.__scale)
        else:
            return exponential.exp(self.log_scale)

    @cache.cached_property
    def log_scale(self):
        if self.__log_scale is not None:
            return chainer.as_variable(self.__log_scale)
        else:
            return exponential.log(self.scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        return ndtr.ndtr((x - self.loc) / self.scale)

    @cache.cached_property
    def entropy(self):
        return self.log_scale + ENTROPYC

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc + self.scale * ndtri.ndtri(x)

    def log_cdf(self, x):
        return log_ndtr.log_ndtr((x - self.loc) / self.scale)

    def log_prob(self, x):
        return (
            LOGPROBC
            - self.log_scale
            - 0.5 * (x - self.loc) ** 2 / self.variance)

    def log_survival_function(self, x):
        return log_ndtr.log_ndtr((self.loc - x) / self.scale)

    @cache.cached_property
    def mean(self):
        return self.loc

    @property
    def params(self):
        return {'loc': self.loc, 'scale': self.scale}

    def prob(self, x):
        return (
            (PROBC / self.scale)
            * exponential.exp(
                -0.5 * (x - self.loc) ** 2 / self.variance))

    def sample_n(self, n):
        dtype = self.loc.dtype
        shape = (n,) + self.loc.shape
        device = self.__device
        if device.xp is cuda.cupy:
            if dtype == numpy.float16:
                # cuRAND supports only FP32 and FP64
                eps = (
                    cuda.cupy.random.standard_normal(
                        shape, dtype=numpy.float32)
                    .astype(numpy.float16))
            else:
                eps = cuda.cupy.random.standard_normal(shape, dtype=dtype)
        elif device.xp is chainerx:
            # TODO(niboshi): Support random in ChainerX
            eps = device.send(
                numpy.random.standard_normal(shape).astype(dtype))
        else:
            eps = numpy.random.standard_normal(shape).astype(dtype)
        return self.loc + self.scale * eps

    @cache.cached_property
    def stddev(self):
        return self.scale

    @property
    def support(self):
        return 'real'

    def survival_function(self, x):
        return ndtr.ndtr((self.loc - x) / self.scale)

    @cache.cached_property
    def variance(self):
        return self.scale ** 2


@distribution.register_kl(Normal, Normal)
def _kl_normal_normal(dist1, dist2):
    return (
        dist2.log_scale
        - dist1.log_scale
        + 0.5 * (dist1.variance + (dist1.loc - dist2.loc)**2) / dist2.variance
        - 0.5)
