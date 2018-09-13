import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import where
from chainer.functions.math import exponential
from chainer.functions.math import lgamma


class FisherSnedecor(distribution.Distribution):

    """FisherSnedecor Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        f(x) = \\frac{1}{B(\\frac{d_1}{2},\\frac{d_2}{2})} \
            \\left(\\frac{d_1}{d_2}\\right)^{\\frac{d_1}{2}} \
            x^{\\frac{d_1}{2}-1} \
            \\left(1+\\frac{d_1}{d_2}x\\right) \
            ^{-\\frac{d_1+d_2}{2}}

    Args:
        d1(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        d2(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, d1, d2):
        super(FisherSnedecor, self).__init__()
        self.__d1 = chainer.as_variable(d1)
        self.__d2 = chainer.as_variable(d2)

    @property
    def d1(self):
        return self.__d1

    @property
    def d2(self):
        return self.__d2

    @property
    def batch_shape(self):
        return self.d1.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.d1.data, cuda.ndarray)

    def log_prob(self, x):
        xp = self.d1.xp
        x = xp.asarray(x)
        sum_d = self.d1 + self.d2
        d1x = self.d1 * x
        return 0.5 * self.d1 * exponential.log(d1x) \
            + 0.5 * self.d2 * exponential.log(self.d2) \
            - 0.5 * (sum_d) \
            * exponential.log(d1x + self.d2) - exponential.log(x) \
            - lgamma.lgamma(0.5 * self.d1) - lgamma.lgamma(0.5 * self.d2) \
            + lgamma.lgamma(0.5 * (sum_d))

    @property
    def mean(self):
        mean_ = self.d2 / (self.d2 - 2)
        xp = mean_.xp
        inf = xp.full_like(mean_.array, xp.inf)
        return where.where(
            xp.asarray(self.d2.data > 2), mean_, xp.asarray(inf))

    def sample_n(self, n):
        xp = self.d1.xp
        if xp is cuda.cupy:
            eps = xp.random.f(
                self.d1.data, self.d2.data, size=(n,) + self.batch_shape,
                dtype=self.d1.dtype)
        else:
            eps = xp.random.f(
                self.d1.data, self.d2.data, size=(n,) + self.batch_shape
            ).astype(self.d1.dtype)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        return 'positive'

    @property
    def variance(self):
        var_ = (2 * self.d2 ** 2 * (self.d1 + self.d2 - 2)
                / self.d1 / (self.d2 - 2) ** 2 / (self.d2 - 4))
        xp = var_.xp
        inf = xp.full_like(var_.array, xp.inf)
        return where.where(
            xp.asarray(self.d2.data > 4), var_, xp.asarray(inf))
