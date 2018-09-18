import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.functions.math import sum


class Multinomial(distribution.Distribution):

    """Multinomial Distribution.

    Args:
        n(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, n, p):
        super(Multinomial, self).__init__()
        self.__n = chainer.as_variable(n)
        self.__p = chainer.as_variable(p)

    @property
    def n(self):
        return self.__n

    @property
    def p(self):
        return self.__p

    @property
    def batch_shape(self):
        return self.p.shape[:-1]

    @property
    def event_shape(self):
        return self.p.shape[-1:]

    @property
    def _is_gpu(self):
        return isinstance(self.p.data, cuda.ndarray)

    def log_prob(self, x):
        if isinstance(x, chainer.Variable):
            x = x.data
        xp = cuda.get_array_module(self.p)
        x = x.astype(self.p.dtype)
        n = self.n.data.astype(self.p.dtype)
        np1 = (n + 1).astype(self.p.dtype)
        return lgamma.lgamma(xp.asarray(np1)) \
            - sum.sum(lgamma.lgamma(x + 1), axis=-1) \
            + sum.sum(x * exponential.log(self.p), axis=-1)

    @property
    def mean(self):
        return self.n * self.p

    def sample_n(self, n):
        xp = cuda.get_array_module(self.p)
        obo_p = self.p.data.reshape(-1, self.p.shape[-1])
        obo_n = xp.broadcast_to(
            self.n.data, self.batch_shape).reshape(-1)
        if xp == cuda.cupy:
            obo_n = obo_n.get()
        print([(one_n, one_p) for one_n, one_p in zip(obo_n, obo_p)])
        eps = [xp.random.multinomial(
            one_n, one_p, size=n) for one_n, one_p
            in zip(obo_n, obo_p)]
        eps = xp.stack(eps)
        eps = xp.swapaxes(eps, 0, 1)
        eps = xp.reshape(eps, (n,)+self.batch_shape+(self.p.shape[-1],))
        noise = chainer.Variable(eps)
        return noise

    @property
    def variance(self):
        n = self.n.data.astype(self.p.dtype)
        return n * self.p * (1. - self.p)
