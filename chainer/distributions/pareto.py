import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.array import where
from chainer.functions.math import exponential


class Pareto(distribution.Distribution):

    """Pareto Distribution.

    .. math::
        f(x) = \\alpha x_m^{\\alpha}(x)^{-(\\alpha+1)},

    Args:
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution :math:`x_m`.
        alpha(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution :math:`\\alpha`.
    """

    def __init__(self, scale, alpha):
        super(Pareto, self).__init__()
        self.__scale = chainer.as_variable(scale)
        self.__alpha = chainer.as_variable(alpha)

    @property
    def scale(self):
        return self.__scale

    @property
    def alpha(self):
        return self.__alpha

    @property
    def batch_shape(self):
        return self.scale.shape

    @property
    def entropy(self):
        return - exponential.log(self.alpha) + exponential.log(self.scale) \
            + 1. / self.alpha + 1.

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.scale.data, cuda.ndarray)

    def log_prob(self, x):
        x = chainer.as_variable(x)
        logp = exponential.log(self.alpha) \
            + self.alpha * exponential.log(self.scale) \
            - (self.alpha + 1) * exponential.log(x)
        xp = logp.xp
        inf = xp.full_like(logp.array, xp.inf)
        return where.where(xp.asarray(x.data >= self.scale.data), logp,
                           xp.asarray(-inf))

    @property
    def mean(self):
        mean = (self.alpha * self.scale / (self.alpha - 1))
        xp = mean.xp
        inf = xp.full_like(mean.array, xp.inf)
        return where.where(self.alpha.data > 1, mean, inf)

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

    @property
    def variance(self):
        var = self.scale ** 2 * self.alpha / (self.alpha - 1) ** 2 \
            / (self.alpha - 2)
        xp = var.xp
        inf = xp.full_like(var.array, xp.inf)
        return where.where(self.alpha.data > 2, var, inf)


@distribution.register_kl(Pareto, Pareto)
def _kl_pareto_pareto(dist1, dist2):
    kl = dist2.alpha * (exponential.log(dist1.scale)
                        - exponential.log(dist2.scale)) \
        + exponential.log(dist1.alpha) - exponential.log(dist2.alpha) \
        + (dist2.alpha - dist1.alpha) / dist1.alpha
    xp = kl.xp
    inf = xp.full_like(kl.array, xp.inf)
    return where.where(dist1.scale.data >= dist2.scale.data, kl, inf)
