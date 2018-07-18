import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import broadcast
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.functions.math import sum as sum_mod
import numpy


class Dirichlet(distribution.Distribution):

    """Dirichlet Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x) = \\frac{\\Gamma(\\sum_{i=1}^{K} \\alpha_i)}
            {\\prod_{i=1}^{K} \\Gamma (\\alpha_i)}
            \\prod_{i=1}^{K} {x_i}^{\\alpha_i-1}

    Args:
        alpha(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, alpha):
        self.__alpha = chainer.as_variable(alpha)
        self.__k = alpha.shape[-1]

    @property
    def alpha(self):
        return self.__alpha

    @property
    def k(self):
        return self.__k

    @property
    def alpha0(self):
        return sum_mod.sum(self.alpha, axis=-1)

    @property
    def batch_shape(self):
        return self.alpha.shape[:-1]

    @property
    def entropy(self):
        return sum_mod.sum(lgamma.lgamma(self.alpha), axis=-1) \
            - lgamma.lgamma(self.alpha0) \
            + (self.alpha0 - self.k) * digamma.digamma(self.alpha0) \
            - sum_mod.sum((self.alpha - 1)
                          * digamma.digamma(self.alpha), axis=-1)

    @property
    def event_shape(self):
        return self.alpha.shape[-1:]

    def log_prob(self, x):
        balpha = broadcast.broadcast_to(self.alpha, x.shape)
        balpha0 = broadcast.broadcast_to(self.alpha0, x.shape[:-1])
        return - sum_mod.sum(lgamma.lgamma(balpha), axis=-1) \
            + lgamma.lgamma(balpha0) \
            + sum_mod.sum((balpha - 1) * exponential.log(x), axis=-1)

    @property
    def mean(self):
        br_alpha0 = expand_dims.expand_dims(self.alpha0, axis=-1)
        br_alpha0 = broadcast.broadcast_to(br_alpha0, self.alpha.shape)
        return self.alpha / br_alpha0

    def sample_n(self, n):
        obo_alpha = self.alpha.data.reshape(-1, self.k)
        xp = cuda.get_array_module(self.alpha)
        # TODO: fix cupy.random.dirichlet to behave same as numpy.
        if xp is numpy:
            eps = [xp.random.dirichlet(
                one_alpha, size=(n,)).astype(numpy.float32)
                for one_alpha in obo_alpha]
        else:
            eps = [xp.random.dirichlet(
                one_alpha, size=(n, self.k)).astype(numpy.float32)
                for one_alpha in obo_alpha]
        eps = xp.swapaxes(xp.stack(eps), 0, 1)
        eps = eps.reshape((n,) + self.alpha.shape)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        return '[0, 1]'

    @property
    def variance(self):
        br_alpha0 = expand_dims.expand_dims(self.alpha0, axis=-1)
        br_alpha0 = broadcast.broadcast_to(br_alpha0, self.alpha.shape)
        return self.alpha * (br_alpha0 - self.alpha) \
            / br_alpha0 ** 2 / (br_alpha0 + 1)


@distribution.register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(dist1, dist2):
    return lgamma.lgamma(dist1.alpha0) \
        - sum_mod.sum(lgamma.lgamma(dist1.alpha), axis=-1) \
        - lgamma.lgamma(dist2.alpha0) \
        + sum_mod.sum(lgamma.lgamma(dist2.alpha), axis=-1) \
        + sum_mod.sum((dist1.alpha - dist2.alpha) * (
            digamma.digamma(dist1.alpha)
            - repeat.repeat(expand_dims.expand_dims(digamma.digamma(
                dist1.alpha0), axis=-1), dist1.k, axis=-1)), axis=-1)
