import math

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer import utils
from chainer.utils import cache


class LaplaceCDF(chainer.function_node.FunctionNode):

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        y = 0.5 - 0.5 * xp.sign(x) * xp.expm1(-abs(x))
        self.retain_outputs((0,))
        return utils.force_array(y, x.dtype),

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        y, = self.get_retained_outputs()
        return (0.5 - abs(y - 0.5)) * gy,


class LaplaceICDF(chainer.function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = backend.get_array_module(x)
        h = 1 - 2 * x
        return utils.force_array(xp.sign(h) * xp.log1p(-abs(h)), x.dtype),

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        x, = self.get_retained_inputs()
        return gy / (0.5 - abs(x - 0.5)),


def _laplace_cdf(x):
    y, = LaplaceCDF().apply((x,))
    return y


def _laplace_icdf(x):
    y, = LaplaceICDF().apply((x,))
    return y


class Laplace(distribution.Distribution):

    """Laplace Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,b) = \\frac{1}{2b}
            \\exp\\left(-\\frac{|x-\\mu|}{b}\\right)

    Args:
        loc(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the location :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter
            of distribution representing the scale :math:`b`.
    """

    def __init__(self, loc, scale):
        super(Laplace, self).__init__()
        self.__loc = loc
        self.__scale = scale

    @cache.cached_property
    def loc(self):
        return chainer.as_variable(self.__loc)

    @cache.cached_property
    def scale(self):
        return chainer.as_variable(self.__scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        return _laplace_cdf((x - self.loc) / self.scale)

    @cache.cached_property
    def entropy(self):
        return 1. + exponential.log(2 * self.scale)

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc + self.scale * _laplace_icdf(x)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        scale = self.scale
        return - exponential.log(2 * scale) - abs(x - self.loc) / scale

    @cache.cached_property
    def mean(self):
        return self.loc

    @cache.cached_property
    def mode(self):
        return self.loc

    @property
    def params(self):
        return {'loc': self.loc, 'scale': self.scale}

    def prob(self, x):
        scale = self.scale
        return 0.5 / scale * exponential.exp(- abs(x - self.loc) / scale)

    def sample_n(self, n):
        if self._is_gpu:
            eps = (
                cuda.cupy.random.laplace(size=(n,) + self.loc.shape)
                .astype(numpy.float32))
        else:
            eps = (
                numpy.random.laplace(size=(n,) + self.loc.shape)
                .astype(numpy.float32))

        return self.scale * eps + self.loc

    @cache.cached_property
    def stddev(self):
        return math.sqrt(2) * self.scale

    @property
    def support(self):
        return 'real'

    @cache.cached_property
    def variance(self):
        return 2 * self.scale ** 2


@distribution.register_kl(Laplace, Laplace)
def _kl_laplace_laplace(dist1, dist2):
    diff = abs(dist1.loc - dist2.loc)
    return (
        exponential.log(dist2.scale)
        - exponential.log(dist1.scale)
        + diff / dist2.scale
        + dist1.scale / dist2.scale * exponential.exp(-diff / dist1.scale)
        - 1)
