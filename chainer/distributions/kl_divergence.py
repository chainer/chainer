_KLDIVERGENCE = {}


def register_kl(dist1, dist2):
    """Decorator to register KL divergence function.

    This decorator registers a function which computes Kullback-Leibler
    divergence. This function will be called by `kl_divergence` based on the
    argument types.

     Args:
         Dist1(`type`): type of a class inherit from `~chainer.Distribution` to
         calculate KL divergence.
         Dist2(`type`): type of a class inherit from `~chainer.Distribution` to
         calculate KL divergence.

    The decorated functoion takes a instance of `Dist1` and `Dist2` and
    returns KL divergence value. When `kl_divergence` will be called taking a
    instance of `Dist1` and `Dist2` after registration, the decorated function
    will be called.

    .. admonition:: Example

       >>> from chainer import distributions
       ... @register_kl(Dist1, Dist2)
       ... def _kl_dist1_dist2(dist1, dist2):
       ...     return KL

    """
    def f(kl):
        _KLDIVERGENCE[dist1, dist2] = kl
    return f


def kl_divergence(dist1, dist2):
    """Computes Kullback-Leibler divergence.

    For two continuous distributions :math:`p(x), q(x)`, it is expressed as

    .. math::
        D_{KL}(p||q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx

    For two discrete distributions :math:`p(x), q(x)`, it is expressed as

    .. math::
        D_{KL}(p||q) = \\sum_x p(x) \\log \\frac{p(x)}{q(x)}

    Args:
        dist1(:class:`~chainer.Distribution`): Distribution to calculate KL
            divergence :math:`p`. This is the first (left) operand of the KL
            divergence.
        dist2(:class:`~chainer.Distribution`): Distribution to calculate KL
            divergence :math:`q`. This is the second (right) operand of the KL
            divergence.

    Returns:
        ~chainer.Variable: Output variable representing kl divergence
            :math:`D_{KL}(p||q)`.

    Using `register_kl`, we can define behavior of `kl_divergence` for any two
    distributions.

    """
    return _KLDIVERGENCE[type(dist1), type(dist2)](dist1, dist2)


def cross_entropy(dist1, dist2):
    """Computes Cross entropy.

    For two continuous distributions :math:`p(x), q(x)`, it is expressed as

    .. math::
        H(p,q) = - \\int p(x) \\log q(x) dx

    For two discrete distributions :math:`p(x), q(x)`, it is expressed as

    .. math::
        H(p,q) = - \\sum_x p(x) \\log q(x)

    This function call `kl_divergence` and `entropy` of `dist1`. Therefore,
    it is necessary to register KL divergence function with `register_kl`
    decoartor and define `entropy` in `dist1`.

    Args:
        dist1(:class:`~chainer.Distribution`): Distribution to calculate cross
            entropy :math:`p`. This is the first (left) operand of the cross
            entropy.
        dist2(:class:`~chainer.Distribution`): Distribution to calculate cross
            entropy :math:`q`. This is the second (right) operand of the cross
            entropy.

    Returns:
        ~chainer.Variable: Output variable representing cross entropy
            :math:`H(p,q)`.

    """
    return dist1.entropy() + kl_divergence(dist1, dist2)
