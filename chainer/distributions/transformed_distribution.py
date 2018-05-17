from chainer import Distribution


class TransformedDistribution(Distribution):

    """Transformed Distribution.

    Args:
        base_distribution(:class:`~chainer.Distribution`): Distribution
        forward(func):
        inv(func):
        inv_logdet_jac(func):

    """

    def __init__(self, base_distribution, forward, inv, inv_logdet_jac):
        self.base_distribution = base_distribution
        self.forward = forward
        self.inv = inv
        self.inv_logdet_jac = inv_logdet_jac

    def cdf(self, x):
        return self.base_distribution.cdf(self.inv(x))

    def log_prob(self, x):
        return self.base_distribution.log_prob(self.inv(x)) \
            + self.inv_logdet_jac(x)

    def sample(self, shape):
        noise = self.base_distribution.sample(shape)
        return self.forward(noise)
