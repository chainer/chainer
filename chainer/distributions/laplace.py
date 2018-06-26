import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.math import exponential
from chainer.functions.math import sign
from chainer import utils
import math
import numpy


class LaplaceCDF(chainer.function_node.FunctionNode):

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        y = 0.5 - 0.5 * xp.sign(x) * xp.expm1(-abs(x))
        self.retain_outputs((0,))
        return utils.force_array(y, x.dtype),

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        y, = self.get_retained_outputs()
        return utils.force_array((0.5 - abs(y - 0.5)) * gy),


class Laplace(distribution.Distribution):

    """Laplace Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,b) = \\frac{1}{2b}
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

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        y, = LaplaceCDF().apply(((x - bl) / bs,))
        return y

    @property
    def entropy(self):
        return 1. + exponential.log(2 * self.scale)

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc - self.scale * sign.sign(x - 0.5) \
            * exponential.log(- abs(2 * x - 1) + 1)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return - exponential.log(2 * bs) - abs(x - bl) / bs

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    def prob(self, x):
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return 0.5 / bs * exponential.exp(- abs(x - bl) / bs)

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
        return math.sqrt(2) * self.scale

    @property
    def support(self):
        return 'real'

    @property
    def variance(self):
        return 2 * self.scale ** 2


@distribution.register_kl(Laplace, Laplace)
def _kl_laplace_laplace(dist1, dist2):
    diff = abs(dist1.loc - dist2.loc)
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + diff / dist2.scale \
        + dist1.scale / dist2.scale * exponential.exp(- diff / dist1.scale) - 1
