import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import where
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma


class StudentT(distribution.Distribution):

    """StudentT Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x) = \\frac{\\Gamma(\\frac{\\nu+1}{2})} \
            {\\sqrt{\\nu\\pi}\\Gamma(\\frac{\\nu}{2})} \
            \\left(1 + \\frac{x^2}{\\nu} \\right)^{-(\\frac{\\nu+1}{2})},

    Args:
        nu(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        degree of freedom :math:`\\nu`.
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale.
    """

    def __init__(self, nu, loc, scale):
        super(StudentT, self).__init__()
        self.__nu = chainer.as_variable(nu)
        self.__loc = chainer.as_variable(loc)
        self.__scale = chainer.as_variable(scale)

    @property
    def nu(self):
        return self.__nu

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
        lgamma05 = 0.57236494292469997
        return exponential.log(self.scale) \
            + 0.5 * (self.nu + 1) * (digamma.digamma(0.5 * (1 + self.nu))
                                     - digamma.digamma(0.5 * self.nu)) \
            + 0.5 * exponential.log(self.nu) \
            + lgamma.lgamma(0.5 * self.nu) + lgamma05 \
            - lgamma.lgamma(0.5 * self.nu + 0.5)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        return lgamma.lgamma(0.5 * (self.nu + 1)) \
            - 0.5 * exponential.log(self.nu * numpy.pi) \
            - lgamma.lgamma(0.5 * self.nu) \
            - (0.5 * (self.nu + 1)) * exponential.log(
                1 + ((x - self.loc)/self.scale) ** 2 / self.nu) \
            - exponential.log(self.scale)

    @property
    def mean(self):
        return self.loc

    def sample_n(self, n):
        xp = cuda.get_array_module(self.nu)
        if xp is cuda.cupy:
            eps = xp.random.standard_t(
                df=cuda.to_cpu(self.nu.data),
                size=(n,)+self.batch_shape, dtype=self.nu.dtype)
        else:
            eps = xp.random.standard_t(
                df=cuda.to_cpu(self.nu.data),
                size=(n,)+self.batch_shape).astype(self.nu.dtype)

        noise = eps * self.scale + self.loc
        return noise

    @property
    def support(self):
        return 'real'

    @property
    def variance(self):
        xp = cuda.get_array_module(self.nu)
        std_var = (self.nu / (self.nu - 2.))
        inf = xp.full_like(std_var.array, xp.inf)
        std_var = where.where(xp.asarray(self.nu.data > 2.), std_var, inf)
        return self.scale ** 2 * std_var
