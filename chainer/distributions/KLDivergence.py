from chainer import distributions
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import exponential
from chainer.functions.math import digamma
from chainer.functions.math import lgamma
from chainer.functions.math import sum

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


@register_kl(distributions.Bernoulli, distributions.Bernoulli)
def _kl_bernoulli_bernoulli(dist1, dist2):
    return dist1.p * (exponential.log(dist1.p) - exponential.log(dist2.p)) \
        + (1 - dist1.p) * (exponential.log(1 - dist1.p)
                           - exponential.log(1 - dist2.p))


@register_kl(distributions.Beta, distributions.Beta)
def _kl_beta_beta(dist1, dist2):
    return - (lgamma.lgamma(dist1.a) + lgamma.lgamma(dist1.b)
              - lgamma.lgamma(dist1.a + dist1.b)) \
        + (lgamma.lgamma(dist2.a) + lgamma.lgamma(dist2.b)
           - lgamma.lgamma(dist2.a + dist2.b)) \
        + (dist1.a - dist2.a) * digamma.digamma(dist1.a) \
        + (dist1.b - dist2.b) * digamma.digamma(dist1.b) \
        + (dist2.a - dist1.a + dist2.b - dist1.b) \
        * digamma.digamma(dist1.a + dist1.b)


@register_kl(distributions.Categorical, distributions.Categorical)
def _kl_categorical_categorical(dist1, dist2):
    return sum.sum(dist1.p * (
        exponential.log(dist1.p) - exponential.log(dist2.p)), axis=-1)


@register_kl(distributions.Dirichlet, distributions.Dirichlet)
def _kl_dirichlet_dirichlet(dist1, dist2):
    return lgamma.lgamma(dist1.alpha0) \
        - sum.sum(lgamma.lgamma(dist1.alpha), axis=-1) \
        - lgamma.lgamma(dist2.alpha0) \
        + sum.sum(lgamma.lgamma(dist2.alpha), axis=-1) \
        + sum.sum((dist1.alpha - dist2.alpha) * (
            digamma.digamma(dist1.alpha)
            - repeat.repeat(expand_dims.expand_dims(digamma.digamma(
                dist1.alpha0), axis=-1), dist1.k, axis=-1)), axis=-1)


@register_kl(distributions.Exponential, distributions.Exponential)
def _kl_exponential_exponential(dist1, dist2):
    return exponential.log(dist1.lam) - exponential.log(dist2.lam) \
        + dist2.lam / dist1.lam - 1.


@register_kl(distributions.Gamma, distributions.Gamma)
def _kl_gamma_gamma(dist1, dist2):
    return (dist1.k - 1.) * digamma.digamma(dist1.k) \
        - exponential.log(dist1.theta) - dist1.k - lgamma.lgamma(dist1.k) \
        + lgamma.lgamma(dist2.k) + dist2.k * exponential.log(dist2.theta) \
        - (dist2.k - 1.) \
        * (digamma.digamma(dist1.k) + exponential.log(dist1.theta)) \
        + dist1.k * dist1.theta / dist2.theta


@register_kl(distributions.Gumbel, distributions.Gumbel)
def _kl_gumbel_gumbel(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + dist1.euler * (dist1.scale / dist2.scale - 1.) \
        + exponential.exp((dist2.loc - dist1.loc) / dist2.scale
                          + lgamma.lgamma(dist1.scale / dist2.scale + 1.)) \
        - 1 + (dist1.loc - dist2.loc) / dist2.scale


@register_kl(distributions.Normal, distributions.Normal)
def _kl_normal_normal(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + 0.5 * (dist1.scale ** 2 + (dist1.loc - dist2.loc) ** 2) \
        / dist2.scale ** 2 - 0.5
