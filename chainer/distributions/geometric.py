import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential


class Geometric(distribution.Distribution):

    """Geometric Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        f(x = k) = p(1-p)^{k-1},

    Args:
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, p):
        super(Geometric, self).__init__()
        self.__p = chainer.as_variable(p)

    @property
    def p(self):
        return self.__p

    @property
    def batch_shape(self):
        return self.p.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.p.data, cuda.ndarray)

    def log_prob(self, x):
        return (x - 1) * exponential.log(1 - self.p) + exponential.log(self.p)

    @property
    def mean(self):
        return 1 / self.p

    def sample_n(self, n):
        xp = cuda.get_array_module(self.p)
        if xp is cuda.cupy:
            eps = xp.random.geometric(
                self.p.data,
                size=(n,)+self.batch_shape, dtype=self.p.dtype)
        else:
            eps = xp.random.geometric(
                self.p.data,
                size=(n,)+self.batch_shape).astype(self.p.dtype)
        return chainer.Variable(eps)

    @property
    def support(self):
        return 'positive integer'

    @property
    def variance(self):
        return (1 - self.p) / self.p ** 2


@distribution.register_kl(Geometric, Geometric)
def _kl_geometric_geometric(dist1, dist2):
    return (1 / dist1.p - 1) \
        * (exponential.log(1 - dist1.p) - exponential.log(1 - dist2.p)) \
        + exponential.log(dist1.p) - exponential.log(dist2.p)
