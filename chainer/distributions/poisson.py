import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer import utils


class Poisson(distribution.Distribution):

    """Poisson Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        P(x; \\lambda) = \\frac{\\lambda ^x e^{-\\lambda}}{x!}

    Args:
        lam(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution. :math:`\\lambda`
    """

    def __init__(self, lam):
        super(Poisson, self).__init__()
        self.__lam = chainer.as_variable(lam)

    @property
    def lam(self):
        return self.__lam

    @property
    def batch_shape(self):
        return self.lam.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.lam.data, cuda.ndarray)

    def log_prob(self, x):
        if isinstance(x, chainer.Variable):
            x = x.data
        x = x.astype(self.lam.dtype)
        xp1 = (x + 1).astype(self.lam.dtype)
        x, xp1 = utils.force_array(x), utils.force_array(xp1)
        return x * exponential.log(self.lam) - lgamma.lgamma(xp1) - self.lam

    @property
    def mean(self):
        return self.lam

    def sample_n(self, n):
        xp = cuda.get_array_module(self.lam)
        if xp is cuda.cupy:
            eps = xp.random.poisson(
                self.lam.data, size=(n,)+self.batch_shape, dtype=xp.float32)
        else:
            eps = xp.random.poisson(
                self.lam.data, size=(n,)+self.batch_shape).astype(xp.float32)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        return 'non negative integer'

    @property
    def variance(self):
        return self.lam


@distribution.register_kl(Poisson, Poisson)
def _kl_poisson_poisson(dist1, dist2):
    return dist1.lam * (exponential.log(dist1.lam)
                        - exponential.log(dist2.lam)) - dist1.lam + dist2.lam
