from chainer import distribution


class TransformedDistribution(distribution.Distribution):

    """Transformed Distribution.

    `TransformedDistribution` is continuous probablity distribution
    transformed from arbitrary continuous distribution by bijective
    (invertible) function. By using this, we can use flexible distribution
    as like Normalizing Flow.

    Args:
        base_distribution(:class:`~chainer.Distribution`): Arbitrary continuous
        distribution.
        bijector(:class:`~chainer.distributions.Bijector`): Bijective
        (invertible) function.
    """

    def __init__(self, base_distribution, bijector):
        self.base_distribution = base_distribution
        self.bijector = bijector

    @property
    def batch_shape(self):
        return self.base_distribution.batch_shape

    @property
    def event_shape(self):
        return self.base_distribution.event_shape

    def cdf(self, x):
        return self.base_distribution.cdf(self.bijector.inv(x))

    def log_prob(self, x):
        return self.base_distribution.log_prob(self.bijector.inv(x)) \
            - self.bijector.logdet_jac(self.bijector.inv(x))

    def sample(self, sample_shape):
        noise = self.base_distribution.sample(sample_shape)
        return self.bijector.forward(noise)
