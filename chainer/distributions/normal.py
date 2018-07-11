import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import erf
from chainer.functions.math import erfinv
from chainer.functions.math import exponential
import math
import numpy


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
        scale :math:`\\sigma`.

    """

    def __init__(self, loc, scale):
        super(Normal, self).__init__()
        self.loc = chainer.as_variable(loc)
        self.scale = chainer.as_variable(scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        return 0.5 * (1. + erf.erf(
            (x - self.loc)
            / (2 ** 0.5 * self.scale)))

    @property
    def entropy(self):
        return exponential.log(self.scale) + ENTROPYC

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc + (
            erfinv.erfinv(2. * x - 1.)
            * (2 ** 0.5 * self.scale))

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_cdf(self, x):
        return exponential.log(self.cdf(x))

    def log_prob(self, x):
        return LOGPROBC - exponential.log(self.scale) \
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2

    def log_survival_function(self, x):
        return exponential.log(self.survival_function(x))

    @property
    def mean(self):
        return self.loc

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
        return 0.5 * (1. - erf.erf(
            (x - self.loc)
            / (2 ** 0.5 * self.scale)))

    @property
    def variance(self):
        return self.scale ** 2


@distribution.register_kl(Normal, Normal)
def _kl_normal_normal(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + 0.5 * (dist1.scale ** 2 + (dist1.loc - dist2.loc) ** 2) \
        / dist2.scale ** 2 - 0.5
