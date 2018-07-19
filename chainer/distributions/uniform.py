import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import where
from chainer.functions.math import clip
from chainer.functions.math import exponential
from chainer.functions.math import sqrt
import numpy


class Uniform(distribution.Distribution):

    """Uniform Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x; l, h) = \\begin{cases}
            \\frac{1}{h - l} (l \\leq x \\leq h) \\\\
            0 (other)
          \\end{cases}

    Args:
        low(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        lower bound :math:`l`.
        high(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        higher bound :math:`h`.
    """

    def __init__(self, low, high):
        self.__low = chainer.as_variable(low)
        self.__high = chainer.as_variable(high)

    @property
    def low(self):
        return self.__low

    @property
    def high(self):
        return self.__high

    @property
    def batch_shape(self):
        return self.low.shape

    def cdf(self, x):
        return clip.clip((x - self.low)/(self.high - self.low), 0., 1.)

    @property
    def entropy(self):
        return exponential.log(self.high - self.low)

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return x * broadcast.broadcast_to(self.high, x.shape) \
            + (1 - x) * broadcast.broadcast_to(self.low, x.shape)

    def log_prob(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)

        bl = broadcast.broadcast_to(self.low, x.shape)
        bh = broadcast.broadcast_to(self.high, x.shape)

        xp = cuda.get_array_module(x)

        logp = -exponential.log(bh - bl)
        return where.where(
            xp.asarray((x.data >= bl.data) & (x.data < bh.data)),
            logp, xp.asarray(-xp.ones_like(x.data)*numpy.inf, dtype=x.dtype))

    @property
    def mean(self):
        return (self.high + self.low) / 2

    def sample_n(self, n):
        xp = cuda.get_array_module(self.low)
        if xp is cuda.cupy:
            eps = xp.random.uniform(
                0, 1, (n,)+self.low.shape, dtype=self.low.dtype)
        else:
            eps = xp.random.uniform(
                0, 1, (n,)+self.low.shape).astype(self.low.dtype)

        noise = self.icdf(eps)

        return noise

    @property
    def stddev(self):
        return sqrt.sqrt(self.variance)

    @property
    def support(self):
        return "[low, high]"

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12


@distribution.register_kl(Uniform, Uniform)
def _kl_uniform_uniform(dist1, dist2):
    xp = cuda.get_array_module(dist1.low)

    is_inf = xp.logical_or(dist1.high.data > dist2.high.data,
                           dist1.low.data < dist2.low.data)
    kl = - exponential.log(dist1.high - dist1.low) \
        + exponential.log(dist2.high - dist2.low)
    inf = xp.asarray(xp.ones_like(dist1.high.data)*numpy.inf,
                     dtype=dist1.high.dtype)

    return where.where(is_inf, inf, kl)
