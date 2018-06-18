import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.math import basic_math
from chainer.functions.math import clip
from chainer.functions.math import exponential
from chainer.functions.math import sign
from chainer.functions.math import sqrt
import numpy


class Laplace(distribution.Distribution):

    """Laplace Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,\\sigma) = \\frac{1}{2b}
            \\exp\\left(-\\frac{|x-\\mu|}{b}\\right)

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`b`.
    """

    def __init__(self, loc, scale):
        super(Laplace, self).__init__()
        self.loc = chainer.as_variable(loc)
        self.scale = chainer.as_variable(scale)

    def __copy__(self):
        return self._copy_to(Laplace(self.loc, self.scale))

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return clip.clip(0.5 * exponential.exp(
            (x - bl) / bs), 0., 0.5) \
            + clip.clip(0.5 - 0.5 * exponential.exp(
                -(x - bl) / bs), 0., 0.5)

    @property
    def entropy(self):
        return 1. + exponential.log(2 * self.scale)

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc - self.scale * sign.sign(x - 0.5) \
            * exponential.log(- basic_math.absolute(2 * x - 1) + 1)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return - exponential.log(2 * bs) \
            - basic_math.absolute(x - bl) / bs

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    def prob(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return 0.5 / bs * exponential.exp(- basic_math.absolute(x - bl) / bs)

    def sample_n(self, n):
        if self._is_gpu:
            eps = cuda.cupy.random.laplace(
                size=(n,) + self.loc.shape).astype(numpy.float32)
        else:
            eps = numpy.random.laplace(
                size=(n,) + self.loc.shape).astype(numpy.float32)

        noise = broadcast.broadcast_to(self.scale, eps.shape) * eps
        noise += broadcast.broadcast_to(self.loc, eps.shape)

        return noise

    @property
    def stddev(self):
        return sqrt.sqrt(2 * self.scale ** 2)

    @property
    def support(self):
        return 'real'

    @property
    def variance(self):
        return 2 * self.scale ** 2


@distribution.register_kl(Laplace, Laplace)
def _kl_laplace_laplace(dist1, dist2):
    diff = basic_math.absolute(dist1.loc - dist2.loc)
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + diff / dist2.scale \
        + dist1.scale / dist2.scale * exponential.exp(- diff / dist1.scale) - 1
