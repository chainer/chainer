import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.activation import sigmoid
from chainer.functions.array import broadcast
from chainer.functions.math import exponential
import numpy


class Bernoulli(distribution.Distribution):

    def __init__(self, p=None, logit=None):
        super(Bernoulli, self).__init__()
        if not (p is None) ^ (logit is None):
            raise ValueError(
                "Either `p` or `logit` (not both) must have a value.")

        with chainer.using_config('enable_backprop', True):
            if p is None:
                self.logit = chainer.as_variable(logit)
                self.p = sigmoid.sigmoid(self.logit)
            else:
                self.p = chainer.as_variable(p)
                self.logit = exponential.log(self.p) - \
                    exponential.log(1. - self.p)

    @property
    def batch_shape(self):
        return self.p.shape

    @property
    def entropy(self):
        return -self.p * exponential.log(self.p) - \
            (numpy.float32(1.) - self.p) \
            * exponential.log(numpy.float32(1.) - self.p)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.p.data, cuda.ndarray)

    def log_prob(self, x):
        if type(x) == chainer.Variable:
            x = x.data
        x = x.astype(self.p.dtype)
        if self._is_gpu:
            invalid_inf = cuda.cupy.zeros_like(x)
            invalid = cuda.cupy.bitwise_and(x != 0, x != 1)
        else:
            invalid_inf = numpy.zeros_like(x)
            invalid = numpy.bitwise_and(x != 0, x != 1)
        invalid_inf[invalid] = numpy.inf

        bl = broadcast.broadcast_to(self.logit, x.shape)
        return bl * (x - 1) - exponential.log(exponential.exp(-bl) + 1)

    @property
    def mean(self):
        return self.p

    def prob(self, x):
        x = chainer.as_variable(x)
        if self._is_gpu:
            valid = cuda.cupy.bitwise_or(x.data == 0, x.data == 1)
        else:
            valid = numpy.bitwise_or(x.data == 0, x.data == 1)
        ret = x * broadcast.broadcast_to(self.p, x.shape) \
            + (1 - x) * (1 - broadcast.broadcast_to(self.p, x.shape))
        return ret * valid

    def sample_n(self, n):
        if self._is_gpu:
            eps = cuda.cupy.random.binomial(
                1, self.p.data, size=(n,)+self.p.shape)
        else:
            eps = numpy.random.binomial(
                1, self.p.data, size=(n,)+self.p.shape)
        return chainer.Variable(eps)

    @property
    def stddev(self):
        return (self.p * (1 - self.p)) ** 0.5

    @property
    def support(self):
        return '{0, 1}'

    @property
    def variance(self):
        return (self.p * (1 - self.p))


@distribution.register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(dist1, dist2):
    return dist1.p * (exponential.log(dist1.p) - exponential.log(dist2.p)) \
        + (1 - dist1.p) * (exponential.log(1 - dist1.p)
                           - exponential.log(1 - dist2.p))
