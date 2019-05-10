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
from chainer.utils import cache


class BernoulliLogProb(chainer.function_node.FunctionNode):

    def __init__(self, binary_check=False):
        super(BernoulliLogProb, self).__init__()
        self.binary_check = binary_check

    def forward(self, inputs):
        logit, x = inputs
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(x)
        y = logit * (x - 1) - xp.log(xp.exp(-logit) + 1)
        y = utils.force_array(y)

        # extreme logit
        logit_isinf = xp.isinf(logit)
        self.logit_ispinf = xp.bitwise_and(logit_isinf, logit > 0)
        self.logit_isminf = xp.bitwise_and(logit_isinf, logit <= 0)
        with numpy.errstate(divide='ignore', invalid='raise'):
            y = xp.where(self.logit_ispinf, xp.log(x), y)
            y = xp.where(self.logit_isminf, xp.log(1 - x), y)

        if self.binary_check:
            self.invalid = utils.force_array(xp.bitwise_and(x != 0, x != 1))
            y[self.invalid] = - xp.inf

        return utils.force_array(y, logit.dtype),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        logit, x = self.get_retained_inputs()
        xp = backend.get_array_module(x)
        dlogit = x - 1. / (1. + exponential.exp(-logit))

        # extreme logit
        nan_dlogit = xp.zeros_like(dlogit.array)
        if self.binary_check:
            nan_dlogit[self.invalid] = numpy.nan
        nan_dlogit[self.logit_ispinf] = numpy.nan
        nan_dlogit[self.logit_isminf] = numpy.nan
        dlogit += nan_dlogit

        return gy * dlogit, None


def _bernoulli_log_prob(logit, x, binary_check=False):
    y, = BernoulliLogProb(binary_check).apply((logit, x))
    return y


class Bernoulli(distribution.Distribution):

    """Bernoulli Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        P(x = 1; p) = p \\\\
        P(x = 0; p) = 1 - p

    Args:
        p(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing :math:`p`. Either `p` or `logit` (not
            both) must have a value.
        logit(:class:`~chainer.Variable` or :ref:`ndarray`) Parameter of
            distribution representing :math:`\\log\\{p/(1-p)\\}`. Either `p`
            or `logit` (not both) must have a value.

    """

    def __init__(self, p=None, logit=None, binary_check=False):
        super(Bernoulli, self).__init__()
        if not (p is None) ^ (logit is None):
            raise ValueError(
                'Either `p` or `logit` (not both) must have a value.')

        self.__p = p
        self.__logit = logit
        self.binary_check = binary_check

    @cache.cached_property
    def p(self):
        if self.__p is not None:
            return chainer.as_variable(self.__p)
        else:
            return sigmoid.sigmoid(self.logit)

    @cache.cached_property
    def logit(self):
        if self.__logit is not None:
            return chainer.as_variable(self.__logit)
        else:
            return exponential.log(self.p) - logarithm_1p.log1p(-self.p)

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
        return _bernoulli_log_prob(self.logit, x, self.binary_check)

    @cache.cached_property
    def mean(self):
        return self.p

    @property
    def params(self):
        return {'logit': self.logit}

    def prob(self, x):
        x = chainer.as_variable(x)
        prob = x * self.p + (1 - x) * (1 - self.p)
        if self.binary_check:
            if self._is_gpu:
                valid = cuda.cupy.bitwise_or(x.array == 0, x.array == 1)
            else:
                valid = numpy.bitwise_or(x.array == 0, x.array == 1)
            prob *= valid
        return prob

    def sample_n(self, n):
        if self._is_gpu:
            eps = cuda.cupy.random.binomial(
                1, self.p.array, size=(n,)+self.p.shape)
        else:
            eps = numpy.random.binomial(
                1, self.p.array, size=(n,)+self.p.shape)
        return chainer.Variable(eps)

    @cache.cached_property
    def stddev(self):
        return self.variance ** 0.5

    @property
    def support(self):
        return '{0, 1}'

    @cache.cached_property
    def variance(self):
        return self.p * (1 - self.p)


@distribution.register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(dist1, dist2):
    return (dist1.logit - dist2.logit) * (dist1.p - 1.) \
        - exponential.log(exponential.exp(-dist1.logit) + 1) \
        + exponential.log(exponential.exp(-dist2.logit) + 1)
