import copy


class Distribution(object):

    """Interface of Distribution

    `Distribution` is a bass class to treat probability distributions.

    This class provides a means to perform following operations.
    1. Sampling random points.
    2. Evaluating a function about probability at given realization value.
        (e.g., probability density function, probability mass function)
    3. Obtaining properties of distributions.
        (e.g., mean, variance)

    Note that every method and property that computes them from
    `chainer.Variable` can basically be differentiated.

    In this class, sampled random points and realization values given
    in function about probability is called "sample". The shape of samples is
    devided into three parts, `sample_shape`, `batch_shape`, and `event_shape`.
    `sample_shape` is the part that is identical and independent. `batch_shape`
    is the part that is not identical and independent. `event_shape` is the
    part that is not identical and dependent.

    When initialization, it takes parameters as inputs. `batch_shape` and
    `event_shape` is decided by the shape of the parameter when generating an
    instance of a class.

    Every function about probability takes realization value whose shape is
    `(sample_shape, batch_shape, event_shape)` and returns evaluated value
    whose shape is `sample_shape`.

    """

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        return target

    @property
    def batch_shape(self):
        """Returns the part of the sample shape that is not identical and independent.

        Returns:
            tuple: The shape of a sample that is not identical and indipendent.

        """
        raise NotImplementedError

    def cdf(self, x):
        """Evaluates the cumulative distribution function at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Cumulative distribution function value evaluated
            at `x`.

        """
        raise NotImplementedError

    @property
    def covariance(self):
        """Returns the covariance of the distribution.

        Returns:
            ~chainer.Variable: The covariance of the distribution.
        """
        raise NotImplementedError

    @property
    def entropy(self):
        """Returns the entropy of the distribution.

        Returns:
            ~chainer.Variable: The entropy of the distribution.

        """
        raise NotImplementedError

    @property
    def event_shape(self):
        """Returns the part of the sample shape that is not identical and dependent.

        Returns:
            tuple: The shape of a sample that is not identical and indipendent.

        """
        raise NotImplementedError

    def icdf(self, x):
        """Evaluates the inverse cumulative distribution function at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Inverse cumulative distribution function value
            evaluated at `x`.

        """
        raise NotImplementedError

    def log_cdf(self, x):
        """Evaluates the log of cumulative distribution function at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Logarithm of cumulative distribution function
            value evaluated at `x`.

        """
        raise NotImplementedError

    def log_prob(self, x):
        """Evaluates the logarithm of probability at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Logarithm of probability evaluated at `x`.

        """
        raise NotImplementedError

    def log_survival_function(self, x):
        """Evaluates the logarithm of survival function at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Logarithm of survival function value evaluated
            at `x`.

        """
        raise NotImplementedError

    @property
    def mean(self):
        """Returns the mean of the distribution.

        Returns:
            ~chainer.Variable: The mean of the distribution.

        """
        raise NotImplementedError

    @property
    def mode(self):
        """Returns the mode of the distribution.

        Returns:
            ~chainer.Variable: The mode of the distribution.

        """
        raise NotImplementedError

    def perplexity(self, x):
        """Evaluates the perplexity function at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Perplexity function value evaluated at `x`.

        """
        raise NotImplementedError

    def prob(self, x):
        """Evaluates probability at a given input.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Probability evaluated at `x`.

        """
        raise NotImplementedError

    def sample(self, sample_shape=()):
        """Samples random points from the distribution.

        Args:
            sample_shape(:class:`tuple` of :class:`int`): Sampling shape.

        Returns:
            ~chainer.Variable: Sampled random points.
        """
        final_shape = self.batch_shape + self.event_shape
        if sample_shape == ():
            n = 1
        elif isinstance(sample_shape, int):
            n = sample_shape
            final_shape = (n,) + final_shape
        else:
            n = 1
            for shape_ in sample_shape:
                n *= shape_
            final_shape = sample_shape + final_shape
        samples = self._sample_n(n)
        return samples.reshape(final_shape)

    def _sample_n(self, n):
        """Samples n random points from the distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: sampled random points.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """Returns the standard deviation of the distribution.

        Returns:
            ~chainer.Variable: The standard deviation of the distribution.

        """
        raise NotImplementedError

    @property
    def support(self):
        """Returns the support of the distribution.

        Returns:
            string: String that means support of this distribution.

        """
        raise NotImplementedError

    def survival_function(self, x):
        """Returns survival function for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): A data points in the domain of the
            distribution

        Returns:
            ~chainer.Variable: Survival function value evaluated at `x`.

        """
        raise NotImplementedError

    @property
    def variance(self):
        """Returns the variance of the distribution.

        Returns:
            ~chainer.Variable: The variance of the distribution.

        """
        raise NotImplementedError
