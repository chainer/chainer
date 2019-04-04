import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import where
from chainer.functions.math import exponential
from chainer import utils
from chainer.utils import cache


class Pareto(distribution.Distribution):

    """Pareto Distribution.

    .. math::
        f(x) = \\alpha x_m^{\\alpha}(x)^{-(\\alpha+1)},

    Args:
        scale(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`x_m`.
        alpha(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution :math:`\\alpha`.
    """

    def __init__(self, scale, alpha):
        super(Pareto, self).__init__()
        self.__scale = scale
        self.__alpha = alpha

    @cache.cached_property
    def scale(self):
        return chainer.as_variable(self.__scale)

    @cache.cached_property
    def alpha(self):
        return chainer.as_variable(self.__alpha)

    @cache.cached_property
    def _log_scale(self):
        return exponential.log(self.scale)

    @cache.cached_property
    def _log_alpha(self):
        return exponential.log(self.alpha)

    @property
    def batch_shape(self):
        return self.scale.shape

    @cache.cached_property
    def entropy(self):
        return - self._log_alpha + self._log_scale \
            + 1. / self.alpha + 1.

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.scale.data, cuda.ndarray)

    def log_prob(self, x):
        x = chainer.as_variable(x)
        logp = self._log_alpha \
            + self.alpha * self._log_scale \
            - (self.alpha + 1) * exponential.log(x)
        xp = logp.xp
        return where.where(
            utils.force_array(x.data >= self.scale.data),
            logp, xp.array(-xp.inf, logp.dtype))

    @cache.cached_property
    def mean(self):
        mean = (self.alpha * self.scale / (self.alpha - 1))
        xp = mean.xp
        return where.where(
            self.alpha.data > 1,
            mean, xp.array(xp.inf, mean.dtype))

    @property
    def params(self):
        return {'scale': self.scale, 'alpha': self.alpha}

    def sample_n(self, n):
        xp = cuda.get_array_module(self.scale)
        if xp is cuda.cupy:
            eps = xp.random.pareto(
                self.alpha.data, (n,)+self.batch_shape, dtype=self.alpha.dtype)
        else:
            eps = xp.random.pareto(
                self.alpha.data, (n,)+self.batch_shape
            ).astype(self.alpha.dtype)

        noise = self.scale * (eps + 1)
        return noise

    @property
    def support(self):
        return '[scale, inf]'

    @cache.cached_property
    def variance(self):
        var = self.scale ** 2 * self.alpha / (self.alpha - 1) ** 2 \
            / (self.alpha - 2)
        xp = var.xp
        return where.where(
            self.alpha.data > 2,
            var, xp.array(xp.inf, var.dtype))


@distribution.register_kl(Pareto, Pareto)
def _kl_pareto_pareto(dist1, dist2):
    kl = dist2.alpha * (dist1._log_scale - dist2._log_scale) \
        + dist1._log_alpha - dist2._log_alpha \
        + (dist2.alpha - dist1.alpha) / dist1.alpha
    xp = kl.xp
    return where.where(
        dist1.scale.data >= dist2.scale.data,
        kl, xp.array(xp.inf, kl.dtype))
