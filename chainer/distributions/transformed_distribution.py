from chainer import Distribution


class TransformedDistribution(Distribution):

    """Transformed Distribution.

    Args:
        base_distribution(:class:`~chainer.Distribution`): Distribution
        transform(:class:`~chainer.distributions.Bijector`):

    """

    def __init__(self, base_distribution, transform):
        self.base_distribution = base_distribution
        self.transform = transform

    def cdf(self, x):
        return self.base_distribution.cdf(self.transform.inv(x))

    def log_prob(self, x):
        return self.base_distribution.log_prob(self.transform.inv(x)) \
            - self.transform.logdet_jac(self.transform.inv(x))

    def sample(self, shape):
        noise = self.base_distribution.sample(shape)
        return self.transform.forward(noise)
