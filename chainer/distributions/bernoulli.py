import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.activation import sigmoid
from chainer.functions.array import broadcast
from chainer.functions.math import exponential
from chainer.functions.math import logarithm_1p
import numpy


class Bernoulli(distribution.Distribution):

    """Bernoulli Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        P(x = 1; p) = p \\\\
        P(x = 0; p) = 1 - p

    Args:
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing \
        :math:`p`. Either `p` or `logit` (not both) must have a value.
        logit(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing \
        :math:`\\log\\{p/(1-p)\\}`. Either `p` or `logit` (not both) must \
        have a value.

    """

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
                self.logit = exponential.log(self.p) \
                    - logarithm_1p.log1p(-self.p)

    @property
    def batch_shape(self):
        return self.p.shape

    @property
    def entropy(self):
        p = self.p
        q = p.dtype.type(1.) - p
        if self._is_gpu:
            zero_entropy = cuda.cupy.bitwise_or(
                self.p.array == 0, self.p.array == 1)
            not_zero_entropy = cuda.cupy.logical_not(zero_entropy)
        else:
            zero_entropy = numpy.bitwise_or(
                self.p.array == 0, self.p.array == 1)
            not_zero_entropy = numpy.logical_not(zero_entropy)
        return - p * exponential.log(p * not_zero_entropy + zero_entropy) \
            - (q) * exponential.log(q * not_zero_entropy + zero_entropy)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.p.array, cuda.ndarray)

    def log_prob(self, x):
        if isinstance(x, chainer.Variable):
            x = x.array
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
            valid = cuda.cupy.bitwise_or(x.array == 0, x.array == 1)
        else:
            valid = numpy.bitwise_or(x.array == 0, x.array == 1)
        ret = x * broadcast.broadcast_to(self.p, x.shape) \
            + (1 - x) * (1 - broadcast.broadcast_to(self.p, x.shape))
        return ret * valid

    def sample_n(self, n):
        if self._is_gpu:
            eps = cuda.cupy.random.binomial(
                1, self.p.array, size=(n,)+self.p.shape)
        else:
            eps = numpy.random.binomial(
                1, self.p.array, size=(n,)+self.p.shape)
        return chainer.Variable(eps)

    @property
    def stddev(self):
        return (self.p * (1 - self.p)) ** 0.5

    @property
    def support(self):
        return '{0, 1}'

    @property
    def variance(self):
        return self.p * (1 - self.p)


@distribution.register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(dist1, dist2):
    return (dist1.logit - dist2.logit) * (dist1.p - 1.) \
        - exponential.log(exponential.exp(-dist1.logit) + 1) \
        + exponential.log(exponential.exp(-dist2.logit) + 1)
