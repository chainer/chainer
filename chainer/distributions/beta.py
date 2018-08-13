import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.functions.array import where


def _lbeta(a, b):
    return lgamma.lgamma(a) + lgamma.lgamma(b) - lgamma.lgamma(a + b)


class Beta(distribution.Distribution):

    def __init__(self, a, b):
        super(Beta, self).__init__()
        self.__a = chainer.as_variable(a)
        self.__b = chainer.as_variable(b)

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def batch_shape(self):
        return self.a.shape

    @property
    def entropy(self):
        apb = self.a + self.b
        return _lbeta(self.a, self.b) \
            - (self.a - 1) * digamma.digamma(self.a) \
            - (self.b - 1) * digamma.digamma(self.b) \
            + (apb - 2) * digamma.digamma(apb)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.a.data, cuda.ndarray)

    def log_prob(self, x):
        x = chainer.as_variable(x)
        xp = cuda.get_array_module(x)

        ba = broadcast.broadcast_to(self.a, x.shape)
        bb = broadcast.broadcast_to(self.b, x.shape)

        logp = (ba - 1) * exponential.log(x) \
            + (bb - 1) * exponential.log(1 - x) \
            - _lbeta(ba, bb)

        inf = xp.ones_like(ba.data) * xp.inf
        return where.where(xp.bitwise_and(x.data >= 0, x.data <= 1),
                           logp, -inf)

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    def sample_n(self, n):
        xp = cuda.get_array_module(self.a)
        eps = xp.random.beta(self.a.data, self.b.data, size=(n,)+self.a.shape)
        noise = chainer.Variable(eps.astype(self.a.dtype))
        return noise

    @property
    def support(self):
        return '[0, 1]'

    @property
    def variance(self):
        apb = self.a + self.b
        return self.a * self.b / apb ** 2 / (apb + 1)


@distribution.register_kl(Beta, Beta)
def _kl_beta_beta(dist1, dist2):
    dist1_apb = dist1.a + dist1.b
    dist2_apb = dist2.a + dist2.b
    return - _lbeta(dist1.a, dist1.b) + _lbeta(dist2.a, dist2.b)\
        + (dist1.a - dist2.a) * digamma.digamma(dist1.a) \
        + (dist1.b - dist2.b) * digamma.digamma(dist1.b) \
        + (dist2_apb - dist1_apb) * digamma.digamma(dist1_apb)
