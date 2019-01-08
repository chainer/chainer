import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import where
from chainer.functions.math import clip
from chainer.functions.math import exponential
from chainer.functions.math import sqrt
from chainer import utils
from chainer.utils import argument


class Uniform(distribution.Distribution):

    """Uniform Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x; l, h) = \\begin{cases}
            \\frac{1}{h - l} & \\text{if }l \\leq x \\leq h \\\\
            0 & \\text{otherwise}
          \\end{cases}

    Args:
        low(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the lower bound :math:`l`.
        high(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the higher bound :math:`h`.
    """

    def __init__(self, **kwargs):
        low, high, loc, scale = None, None, None, None
        if kwargs:
            low, high, loc, scale = argument.parse_kwargs(
                kwargs, ('low', low), ('high', high), ('loc', loc),
                ('scale', scale))
        if not (low is None or high is None) ^ (loc is None or scale is None):
            raise ValueError(
                "Either `low, high` or `loc, scale` (not both) must have a "
                "value.")
        with chainer.using_config('enable_backprop', True):
            if low is None:
                self.__loc = chainer.as_variable(loc)
                self.__scale = chainer.as_variable(scale)
                self.__low = self.__loc
                self.__high = self.__loc + self.__scale
            else:
                self.__low = chainer.as_variable(low)
                self.__high = chainer.as_variable(high)
                self.__loc = self.__low
                self.__scale = self.__high - self.__low

    @property
    def low(self):
        return self.__low

    @property
    def high(self):
        return self.__high

    @property
    def loc(self):
        return self.__loc

    @property
    def scale(self):
        return self.__scale

    @property
    def batch_shape(self):
        return self.low.shape

    def cdf(self, x):
        return clip.clip((x - self.loc) / self.scale, 0., 1.)

    @property
    def entropy(self):
        return exponential.log(self.scale)

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return x * self.scale + self.loc

    def log_prob(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)

        xp = backend.get_array_module(x)

        logp = broadcast.broadcast_to(
            -exponential.log(self.scale), x.shape)
        return where.where(
            utils.force_array(
                (x.data >= self.low.data) & (x.data <= self.high.data)),
            logp, xp.array(-xp.inf, logp.dtype))

    @property
    def mean(self):
        return (self.high + self.low) / 2

    def sample_n(self, n):
        xp = backend.get_array_module(self.low)
        if xp is cuda.cupy:
            eps = xp.random.uniform(
                0, 1, (n,) + self.low.shape, dtype=self.low.dtype)
        else:
            eps = xp.random.uniform(
                0, 1, (n,) + self.low.shape).astype(self.low.dtype)

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
        return self.scale ** 2 / 12


@distribution.register_kl(Uniform, Uniform)
def _kl_uniform_uniform(dist1, dist2):
    xp = backend.get_array_module(dist1.low)

    is_inf = xp.logical_or(dist1.high.data > dist2.high.data,
                           dist1.low.data < dist2.low.data)
    kl = - exponential.log(dist1.high - dist1.low) \
        + exponential.log(dist2.high - dist2.low)
    inf = xp.array(xp.inf, dist1.high.dtype)

    return where.where(is_inf, inf, kl)
