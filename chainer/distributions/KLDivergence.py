import chainer
from chainer.backends import cuda
from chainer import distributions
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.array import rollaxis
from chainer.functions.math import basic_math
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import inv
from chainer.functions.math import lgamma
from chainer.functions.math import matmul
from chainer.functions.math import sum
import numpy

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
        return kl
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


@register_kl(distributions.Geometric, distributions.Geometric)
def _kl_geometric_geometric(dist1, dist2):
    return (1 / dist1.p - 1)\
        * (exponential.log(1 - dist1.p) - exponential.log(1 - dist2.p)) \
        + exponential.log(dist1.p) - exponential.log(dist2.p)


@register_kl(distributions.Gumbel, distributions.Gumbel)
def _kl_gumbel_gumbel(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + dist1.euler * (dist1.scale / dist2.scale - 1.) \
        + exponential.exp((dist2.loc - dist1.loc) / dist2.scale
                          + lgamma.lgamma(dist1.scale / dist2.scale + 1.)) \
        - 1 + (dist1.loc - dist2.loc) / dist2.scale


@register_kl(distributions.Laplace, distributions.Laplace)
def _kl_laplace_laplace(dist1, dist2):
    diff = basic_math.absolute(dist1.loc - dist2.loc)
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + diff / dist2.scale \
        + dist1.scale / dist2.scale * exponential.exp(- diff / dist1.scale) - 1


@register_kl(distributions.MultivariateNormal,
             distributions.MultivariateNormal)
def _kl_multivariatenormal_multivariatenormal(dist1, dist2):
    st = rollaxis.rollaxis(dist1.scale_tril, -2, 0)
    st = rollaxis.rollaxis(st, -1, 1)
    diag = st[list(range(dist1.d)), list(range(dist1.d))]
    logdet1 = sum.sum(exponential.log(basic_math.absolute(diag)), axis=0)

    st = rollaxis.rollaxis(dist2.scale_tril, -2, 0)
    st = rollaxis.rollaxis(st, -1, 1)
    diag = st[list(range(dist2.d)), list(range(dist2.d))]
    logdet2 = sum.sum(exponential.log(basic_math.absolute(diag)), axis=0)

    scale_tril_inv2 = inv.batch_inv(dist2.scale_tril.reshape(
        -1, dist2.d, dist2.d))
    trace = sum.sum(matmul.matmul(
        scale_tril_inv2, dist1.scale_tril.reshape(-1, dist2.d, dist2.d)) ** 2,
        axis=(-1, -2)).reshape(dist1.batch_shape)

    mu = dist1.loc - dist2.loc
    mah = matmul.matmul(scale_tril_inv2, mu.reshape(-1, dist1.d, 1))
    mah = sum.sum(mah ** 2, axis=-2).reshape(dist1.batch_shape)
    return logdet2 - logdet1 + 0.5 * trace + 0.5 * mah - 0.5 * dist1.d


@register_kl(distributions.Normal, distributions.Normal)
def _kl_normal_normal(dist1, dist2):
    return exponential.log(dist2.scale) - exponential.log(dist1.scale) \
        + 0.5 * (dist1.scale ** 2 + (dist1.loc - dist2.loc) ** 2) \
        / dist2.scale ** 2 - 0.5


@register_kl(distributions.Pareto, distributions.Pareto)
def _kl_pareto_pareto(dist1, dist2):
    return dist2.alpha * (exponential.log(dist1.scale)
                          - exponential.log(dist2.scale)) \
        + exponential.log(dist1.alpha) - exponential.log(dist2.alpha) \
        + (dist2.alpha - dist1.alpha) / dist1.alpha


@register_kl(distributions.Poisson, distributions.Poisson)
def _kl_poisson_poisson(dist1, dist2):
    return dist1.lam * (exponential.log(dist1.lam)
                        - exponential.log(dist2.lam)) - dist1.lam + dist2.lam


@register_kl(distributions.Uniform, distributions.Uniform)
def _kl_uniform_uniform(dist1, dist2):
    if dist1._is_gpu:
        is_inf = cuda.cupy.logical_or(dist1.high.data > dist2.high.data,
                                      dist1.low.data < dist2.low.data)
        inf = cuda.cupy.zeros_like(dist1.high.data)
        inf[is_inf] = numpy.inf
    else:
        is_inf = numpy.logical_or(dist1.high.data > dist2.high.data,
                                  dist1.low.data < dist2.low.data)
        inf = numpy.zeros_like(dist1.high.data)
        inf[is_inf] = numpy.inf

    return - exponential.log(dist1.high - dist1.low) \
        + exponential.log(dist2.high - dist2.low) + inf


@register_kl(distributions.Bernoulli, distributions.Poisson)
def _kl_bernoulli_poisson(dist1, dist2):
    return dist1.p * (exponential.log(dist1.p) - exponential.log(dist2.lam)) \
        + (1 - dist1.p) * exponential.log(1 - dist1.p) + dist2.lam


@register_kl(distributions.Beta, distributions.Pareto)
def _kl_beta_pareto(dist1, dist2):
    if dist1._is_gpu:
        inf = cuda.cupy.ones_like(dist1.a.data) * numpy.inf
    else:
        inf = numpy.ones_like(dist1.a.data) * numpy.inf
    return chainer.Variable(inf)


@register_kl(distributions.Beta, distributions.Exponential)
def _kl_beta_exponential(dist1, dist2):
    return - dist1.entropy - exponential.log(dist2.lam) \
        + dist2.lam * dist1.a / (dist1.a + dist1.b)


@register_kl(distributions.Beta, distributions.Gamma)
def _kl_beta_gamma(dist1, dist2):
    return - dist1.entropy + lgamma.lgamma(dist2.k) \
        + dist2.k * exponential.log(dist2.theta) \
        - (dist2.k - 1) * (digamma.digamma(dist1.a)
                           - digamma.digamma(dist1.a + dist1.b)) \
        + dist1.a / (dist1.a + dist1.b) / dist2.theta


@register_kl(distributions.Beta, distributions.Normal)
def _kl_beta_normal(dist1, dist2):
    apb = dist1.a + dist1.b
    return - dist1.entropy + 0.5 * numpy.log(2 * numpy.pi) \
        + exponential.log(dist2.scale) \
        + (dist1.a * (dist1.a + 1) / apb / (apb + 1) / 2
           - dist2.loc * dist1.a / apb
           + dist2.loc ** 2 / 2) / dist2.scale ** 2


@register_kl(distributions.Beta, distributions.Uniform)
def _kl_beta_uniform(dist1, dist2):
    if dist1._is_gpu:
        is_inf = cuda.cupy.logical_or(dist2.high.data < 1,
                                      dist2.low.data > 0)
        inf = cuda.cupy.zeros_like(dist1.a.data)
        inf[is_inf] = numpy.inf
    else:
        is_inf = numpy.logical_or(dist2.high.data < 1,
                                  dist2.low.data > 0)
        inf = numpy.zeros_like(dist1.a.data)
        inf[is_inf] = numpy.inf
    return - dist1.entropy + exponential.log(dist2.high - dist2.low) + inf


@register_kl(distributions.Exponential, distributions.Beta)
@register_kl(distributions.Exponential, distributions.Pareto)
@register_kl(distributions.Exponential, distributions.Uniform)
def _kl_exponential_inf(dist1, dist2):
    if dist1._is_gpu:
        inf = cuda.cupy.ones_like(dist1.lam.data) * numpy.inf
    else:
        inf = numpy.ones_like(dist1.lam.data) * numpy.inf
    return chainer.Variable(inf)
