import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import distribution
import chainer.distributions.utils
from chainer.functions.activation import sigmoid
from chainer.functions.math import exponential
from chainer.functions.math import logarithm_1p
from chainer import utils


class BernoulliLogProb(chainer.function_node.FunctionNode):

    def forward(self, inputs):
        logit, x = inputs
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(x)
        y = logit * (x - 1) - xp.log(xp.exp(-logit) + 1)
        y = utils.force_array(y)

        self.invalid = utils.force_array(xp.bitwise_and(x != 0, x != 1))
        y[self.invalid] = - xp.inf

        # extreme logit
        logit_isinf = xp.isinf(logit)
        self.to_zero = xp.bitwise_and(
            logit_isinf, xp.sign(x-0.5) == xp.sign(logit))
        self.to_m_inf = xp.bitwise_and(
            logit_isinf, xp.sign(x-0.5) != xp.sign(logit))
        y[self.to_zero] = 0.
        y[self.to_m_inf] = - xp.inf

        return utils.force_array(y, logit.dtype),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        logit, x = self.get_retained_inputs()
        xp = backend.get_array_module(x)
        dlogit = x - 1. / (1. + exponential.exp(-logit))

        # extreme logit
        nan_dlogit = xp.zeros_like(dlogit.array)
        nan_dlogit[self.invalid] = xp.nan
        nan_dlogit[self.to_zero] = xp.nan
        nan_dlogit[self.to_m_inf] = xp.nan
        dlogit += nan_dlogit

        return gy * dlogit, None


def _bernoulli_log_prob(logit, x):
    y, = BernoulliLogProb().apply((logit, x))
    return y


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
        return - chainer.distributions.utils._modified_xlogx(p) \
            - chainer.distributions.utils._modified_xlogx(q)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.p.array, cuda.ndarray)

    def log_prob(self, x):
        return _bernoulli_log_prob(self.logit, x)

    @property
    def mean(self):
        return self.p

    def prob(self, x):
        x = chainer.as_variable(x)
        if self._is_gpu:
            valid = cuda.cupy.bitwise_or(x.array == 0, x.array == 1)
        else:
            valid = numpy.bitwise_or(x.array == 0, x.array == 1)
        ret = x * self.p + (1 - x) * (1 - self.p)
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
