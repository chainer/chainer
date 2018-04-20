class Distribution(object):
    """
    Interface of Distribution.

    `Distribution` is a class to treat probability distribution.
    When Initialization, it takes parameter as input.
    """

    def __init__(self):
        self.event_shape = None
        self.batch_shape = None
        self.enumerate_support = None
        self.support = None
    def log_prob(self, x):
        """

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.
        
        """
        raise NotImplementedError

    def sample(self):
        """

        Returns:
            Output variable representing sampled random variable.
        """
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def covariance(self):
        raise NotImplementedError

    def log_cdf(self, x):
        raise NotImplementedError

    def log_survival_function(self, x):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def perplexity(self, x):
        raise NotImplementedError

    def prob(self, x):
        raise NotImplementedError

    def icdf(self, x):
        raise NotImplementedError

    def stddev(self):
        raise NotImplementedError

    def survival_function(self, x):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError
