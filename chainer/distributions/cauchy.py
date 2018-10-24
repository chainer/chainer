import numpy
import warnings

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import trigonometric
from chainer import utils


class CauchyICDF(chainer.function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = cuda.get_array_module(x)
        self.h = (x - 0.5) * numpy.pi
        return utils.force_array(xp.tan(self.h), x.dtype),

    def backward(self, target_input_indexes, grad_outputs):
        gy, = grad_outputs
        x, = self.get_retained_inputs()
        return gy / trigonometric.cos(
            utils.force_array(self.h, x.dtype)) ** 2 * numpy.pi,


def _cauchy_icdf(x):
    y, = CauchyICDF().apply((x,))
    return y


class Cauchy(distribution.Distribution):

    """Cauchy Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;x_0,\\gamma) = \\frac{1}{\\pi}\\frac{\\gamma}{(x-x_0)^2+\\gamma^2}

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\x_0`.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\gamma`.
    """

    def __init__(self, loc, scale):
        super(Cauchy, self).__init__()
        self.loc = chainer.as_variable(loc)
        self.scale = chainer.as_variable(scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def cdf(self, x):
        return 1 / numpy.pi * trigonometric.arctan(
            (x - self.loc) / self.scale) + 0.5

    @property
    def entropy(self):
        return exponential.log(4 * numpy.pi * self.scale)

    @property
    def event_shape(self):
        return ()

    def icdf(self, x):
        return self.loc + self.scale * _cauchy_icdf(x)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        return - numpy.log(numpy.pi) + exponential.log(self.scale) \
            - exponential.log((x - self.loc)**2 + self.scale**2)

    @property
    def mean(self):
        warnings.warn("Mean of the cauchy distribution is undefined.",
                      RuntimeWarning)
        xp = cuda.get_array_module(self.loc)
        return chainer.as_variable(xp.full_like(self.loc.data, xp.nan))

    def sample_n(self, n):
        xp = cuda.get_array_module(self.loc)
        if xp is cuda.cupy:
            eps = xp.random.standard_cauchy(
                (n,)+self.loc.shape, dtype=self.loc.dtype)
        else:
            eps = xp.random.standard_cauchy(
                (n,)+self.loc.shape).astype(self.loc.dtype)

        noise = self.scale * eps + self.loc

        return noise

    @property
    def support(self):
        return 'real'

    @property
    def variance(self):
        warnings.warn("Variance of the cauchy distribution is undefined.",
                      RuntimeWarning)
        xp = cuda.get_array_module(self.loc)
        return chainer.as_variable(xp.full_like(self.loc.data, xp.nan))
