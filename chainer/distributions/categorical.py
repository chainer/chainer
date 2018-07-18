import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import expand_dims
from chainer.functions.math import exponential
from chainer.functions.math import logsumexp
from chainer.functions.math import sum as sum_mod
from chainer.utils import argument
import numpy


class Categorical(distribution.Distribution):

    """Categorical Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        P(x = i; p) = p_i

    Args:
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        logit(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing \
        :math:`\\log\\{p\\} + C`. Either `p` or `logit` (not both) must \
        have a value.

    """

    def __init__(self, p=None, **kwargs):
        logit = None
        if kwargs:
            logit, = argument.parse_kwargs(
                kwargs, ('logit', logit))
        if not (p is None) ^ (logit is None):
            raise ValueError(
                "Either `p` or `logit` (not both) must have a value.")

        with chainer.using_config('enable_backprop', True):
            if p is None:
                logit = chainer.as_variable(logit)
                self.__log_p = logit - expand_dims.expand_dims(
                    logsumexp.logsumexp(logit, axis=-1), axis=-1)
                self.__p = exponential.exp(self.__log_p)
            else:
                self.__p = chainer.as_variable(p)
                self.__log_p = exponential.log(self.__p)

    @property
    def p(self):
        return self.__p

    @property
    def log_p(self):
        return self.__log_p

    @property
    def batch_shape(self):
        return self.p.shape[:-1]

    @property
    def event_shape(self):
        return ()

    @property
    def entropy(self):
        print(self.p)
        return - sum_mod.sum(
            chainer.distributions.utils._modified_xlogx(self.p), axis=-1)

    def log_prob(self, x):
        mg = numpy.meshgrid(
            *tuple(range(i) for i in self.batch_shape), indexing='ij')
        if isinstance(x, chainer.Variable):
            return self.log_p[mg + [x.data.astype(numpy.int32)]]
        else:
            return self.log_p[mg + [x.astype(numpy.int32)]]

    def sample_n(self, n):
        xp = cuda.get_array_module(self.p)
        onebyone_p = self.p.data.reshape(-1, self.p.shape[-1])
        eps = [xp.random.choice(
            one_p.shape[0], size=(n,), p=one_p) for one_p in onebyone_p]
        eps = xp.vstack(eps).T.reshape((n,)+self.batch_shape)
        noise = chainer.Variable(eps)
        return noise


@distribution.register_kl(Categorical, Categorical)
def _kl_categorical_categorical(dist1, dist2):
    return sum_mod.sum(dist1.p * (
        exponential.log(dist1.p) - exponential.log(dist2.p)), axis=-1)
