import numpy

import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import expand_dims
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.functions.math import sum as sum_mod
from chainer.utils import cache


def _lbeta(x):
    return sum_mod.sum(lgamma.lgamma(x), axis=-1) \
        - lgamma.lgamma(sum_mod.sum(x, axis=-1))


class Dirichlet(distribution.Distribution):

    """Dirichlet Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x) = \\frac{\\Gamma(\\sum_{i=1}^{K} \\alpha_i)}
            {\\prod_{i=1}^{K} \\Gamma (\\alpha_i)}
            \\prod_{i=1}^{K} {x_i}^{\\alpha_i-1}

    Args:
        alpha(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
    """

    def __init__(self, alpha):
        self.__alpha = alpha

    @cache.cached_property
    def alpha(self):
        return chainer.as_variable(self.__alpha)

    @cache.cached_property
    def alpha0(self):
        return sum_mod.sum(self.alpha, axis=-1)

    @property
    def batch_shape(self):
        return self.alpha.shape[:-1]

    @cache.cached_property
    def entropy(self):
        return _lbeta(self.alpha) \
            + (self.alpha0 - self.event_shape[0]) \
            * digamma.digamma(self.alpha0) \
            - sum_mod.sum((self.alpha - 1)
                          * digamma.digamma(self.alpha), axis=-1)

    @property
    def event_shape(self):
        return self.alpha.shape[-1:]

    def log_prob(self, x):
        return - _lbeta(self.alpha) \
            + sum_mod.sum((self.alpha - 1) * exponential.log(x), axis=-1)

    @cache.cached_property
    def mean(self):
        alpha0 = expand_dims.expand_dims(self.alpha0, axis=-1)
        return self.alpha / alpha0

    @property
    def params(self):
        return {'alpha': self.alpha}

    def sample_n(self, n):
        obo_alpha = self.alpha.data.reshape(-1, self.event_shape[0])
        xp = cuda.get_array_module(self.alpha)
        if xp is numpy:
            eps = [xp.random.dirichlet(
                one_alpha, size=(n,)).astype(numpy.float32)
                for one_alpha in obo_alpha]
        else:
            eps = [xp.random.dirichlet(
                one_alpha, size=(n,)).astype(numpy.float32)
                for one_alpha in obo_alpha]
        eps = [xp.expand_dims(eps_, 0) for eps_ in eps]
        eps = xp.swapaxes(xp.vstack(eps), 0, 1)
        eps = eps.reshape((n,) + self.alpha.shape)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        return '[0, 1]'

    @cache.cached_property
    def variance(self):
        alpha0 = expand_dims.expand_dims(self.alpha0, axis=-1)
        return self.alpha * (alpha0 - self.alpha) \
            / alpha0 ** 2 / (alpha0 + 1)


@distribution.register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(dist1, dist2):
    return - _lbeta(dist1.alpha) + _lbeta(dist2.alpha) \
        + sum_mod.sum((dist1.alpha - dist2.alpha) * (
            digamma.digamma(dist1.alpha)
            - expand_dims.expand_dims(digamma.digamma(
                dist1.alpha0), axis=-1)), axis=-1)
