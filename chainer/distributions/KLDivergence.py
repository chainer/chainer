from chainer import distributions
from chainer.functions.math import exponential

_KLDIVERGENCE = {}


def register_kl(dist1, dist2):
    """Decorator to register KL divergence function.

    .. admonition:: Example

       This test case runs only when `numpy>=1.10` is installed.

       >>> from chainer import distributions
       ... @register_kl(distributions.Normal, distributions.Normal)
       ... def _kl_dist1_dist2(dist1, dist2):
       ...     return KL

    Args:
        dist1(:class:`~chainer.Distribution`): Distribution to calculate KL
            divergence.
        dist2(:class:`~chainer.Distribution`): Distribution to calculate KL
            divergence.

    """
    def f(kl):
        _KLDIVERGENCE[dist1, dist2] = kl
    return f


def kl_divergence(dist1, dist2):
    """Kullback–Leibler divergence.

    Args:
        dist1(:class:`~chainer.Distribution`): Distribution to calculate KL
            divergence.
        dist2(:class:`~chainer.Distribution`): Distribution to calculate KL
            divergence.

    Returns:
        ~chainer.Variable: Output variable representing kl divergence
            :math:`D_{KL}(dist1 || dist2)`.

    """
    return _KLDIVERGENCE[type(dist1), type(dist2)](dist1, dist2)


def cross_entropy(dist1, dist2):
    """Cross entropy.

    Args:
        dist1(:class:`~chainer.Distribution`): Distribution to calculate cross
            entropy.
        dist2(:class:`~chainer.Distribution`): Distribution to calculate cross
            entropy.

    Returns:
        ~chainer.Variable: Output variable representing cross entropy
            :math:`H(dist1, dist2)`.

    """
    return dist1.entropy() + kl_divergence(dist1, dist2)


@register_kl(distributions.Normal, distributions.Normal)
def _kl_normal_normal(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + 0.5 * (dist1.scale ** 2 + (dist1.loc - dist2.loc) ** 2) \
        / dist2.scale ** 2 - 0.5
