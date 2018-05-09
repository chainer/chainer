from chainer import distributions
from chainer.functions.math import exponential

_KLDIVERGENCE = {}


def register_kl(dist1, dist2):
    def f(kl):
        _KLDIVERGENCE[dist1, dist2] = kl
    return f


def kl_divergence(dist1, dist2):
    return _KLDIVERGENCE[type(dist1), type(dist2)](dist1, dist2)


@register_kl(distributions.Normal, distributions.Normal)
def _kl_normal_normal(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + 0.5 * (dist1.scale ** 2 + (dist1.loc - dist2.loc) ** 2) \
        / dist2.scale ** 2 - 0.5
