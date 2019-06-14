import numpy

import chainer
from chainer import backend
from chainer import distribution
from chainer.functions.activation import log_softmax
from chainer.functions.math import exponential
from chainer.functions.math import sum as sum_mod
from chainer.utils import argument
from chainer.utils import cache


class Categorical(distribution.Distribution):

    """Categorical Distribution.

    The probability mass function of the distribution is expressed as

    .. math::
        P(x = i; p) = p_i

    Args:
        p(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
        logit(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing :math:`\\log\\{p\\} + C`. Either `p` or
            `logit` (not both) must have a value.

    """

    def __init__(self, p=None, **kwargs):
        logit = None
        if kwargs:
            logit, = argument.parse_kwargs(
                kwargs, ('logit', logit))
        if not (p is None) ^ (logit is None):
            raise ValueError(
                'Either `p` or `logit` (not both) must have a value.')
        self.__p = p
        self.__logit = logit

    @cache.cached_property
    def p(self):
        if self.__p is not None:
            return chainer.as_variable(self.__p)
        else:
            return exponential.exp(self.log_p)

    @cache.cached_property
    def log_p(self):
        if self.__p is not None:
            return exponential.log(self.__p)
        else:
            return log_softmax.log_softmax(self.__logit, axis=-1)

    @property
    def batch_shape(self):
        return self.p.shape[:-1]

    @property
    def event_shape(self):
        return ()

    @cache.cached_property
    def entropy(self):
        return - sum_mod.sum(
            chainer.distributions.utils._modified_xlogx(self.p), axis=-1)

    def log_prob(self, x):
        mg = numpy.meshgrid(
            *tuple(range(i) for i in self.batch_shape), indexing='ij')
        if isinstance(x, chainer.Variable):
            return self.log_p[mg + [x.data.astype(numpy.int32)]]
        else:
            return self.log_p[mg + [x.astype(numpy.int32)]]

    @property
    def params(self):
        return {'p': self.p}

    def sample_n(self, n):
        xp = backend.get_array_module(self.p)
        onebyone_p = self.p.data.reshape(-1, self.p.shape[-1])
        eps = [xp.random.choice(
            one_p.shape[0], size=(n,), p=one_p) for one_p in onebyone_p]
        eps = xp.vstack(eps).T.reshape((n,)+self.batch_shape)
        noise = chainer.Variable(eps)
        return noise


@distribution.register_kl(Categorical, Categorical)
def _kl_categorical_categorical(dist1, dist2):
    return sum_mod.sum(dist1.p * (dist1.log_p - dist2.log_p), axis=-1)
