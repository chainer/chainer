import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.math import exponential
import math

LOGPROBC = - 0.5 * math.log(2 * math.pi)


class LogNormal(distribution.Distribution):

    """Logatithm Normal Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,\\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}x}
            \\exp\\left(-\\frac{(\\log{x}-\\mu)^2}{2\\sigma^2}\\right)

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution :math:`\\sigma`.
    """

    def __init__(self, loc, scale):
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
        return 0.5 - LOGPROBC + exponential.log(self.scale) + self.loc

    @property
    def event_shape(self):
        return ()

    def log_prob(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return LOGPROBC - exponential.log(bs) \
            - exponential.log(x) \
            - (0.5 * (exponential.log(x) - bl) ** 2 / bs ** 2)

    @property
    def mean(self):
        return exponential.exp(self.loc + 0.5 * self.scale ** 2)

    def sample_n(self, n):
        xp = cuda.get_array_module(self.loc)
        if xp is cuda.cupy:
            eps = xp.random.standard_normal(
                (n,)+self.loc.shape, dtype=self.loc.dtype)
        else:
            eps = xp.random.standard_normal(
                (n,)+self.loc.shape).astype(self.loc.dtype)

        noise = broadcast.broadcast_to(self.scale, eps.shape) * eps
        noise += broadcast.broadcast_to(self.loc, eps.shape)

        return exponential.exp(noise)

    @property
    def support(self):
        return 'positive'

    @property
    def variance(self):
        return exponential.exp(2 * self.loc + self.scale ** 2) \
            * (exponential.exp(self.scale ** 2) - 1)


@distribution.register_kl(LogNormal, LogNormal)
def _kl_log_normal_log_normal(dist1, dist2):
    return 0.5 * ((dist1.loc - dist2.loc) ** 2 +
                  dist1.scale ** 2) / dist2.scale ** 2 - 0.5 \
        + exponential.log(dist2.scale) - exponential.log(dist1.scale)
