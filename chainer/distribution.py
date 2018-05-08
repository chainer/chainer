import copy


class Distribution(object):

    """Interface of Distribution

    `Distribution` is a bass class to treat probability distributions.
    When initialization, it takes parameter as input.
    """

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        return target

    @property
    def batch_shape(self):
        """Returns the shape of a sample.

        Returns:
            ~chainer.Variable: Output variable representing the shape of a
            sample.

        """
        raise NotImplementedError

    def cdf(self, x):
        """Returns Cumulative Distribution Function for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing Cumulative
            Distribution Function.

        """
        raise NotImplementedError

    @property
    def covariance(self):
        """Returns covariance.

        Returns:
            ~chainer.Variable: Output variable representing covariance.
        """
        raise NotImplementedError

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            ~chainer.Variable: Output variable representing entropy.

        """
        raise NotImplementedError

    @property
    def event_shape(self):
        """Returns the shape of an event.

        Returns:
            ~chainer.Variable: Output variable representing the shape of an
            event.

        """
        raise NotImplementedError

    def icdf(self, x):
        """Returns Inverse Cumulative Distribution Function for a input Variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing Inverse Cumulative
            Distribution Function.

        """
        raise NotImplementedError

    def log_cdf(self, x):
        """Returns logarithm of Cumulative Distribution Function for a input Variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing logarithm of
            Cumulative Distribution Function.

        """
        raise NotImplementedError

    def log_prob(self, x):
        """Returns logarithm of probability for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing logarithm of
            probability.

        """
        raise NotImplementedError

    def log_survival_function(self, x):
        """Returns logarithm of survival function for a input Variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing logarithm of
            survival function for a input variable.

        """
        raise NotImplementedError

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        raise NotImplementedError

    @property
    def mode(self):
        """Returns mode.

        Returns:
            ~chainer.Variable: Output variable representing mode.

        """
        raise NotImplementedError

    def perplexity(self, x):
        """Returns perplexity function for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing perplexity function
            for a input variable.

        """
        raise NotImplementedError

    def prob(self, x):
        """Returns probability for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing probability.

        """
        raise NotImplementedError

    def sample(self, shape=()):
        """Samples from this distribution.

        Args:
            shape(:class:`tuple` of :class:`int`): Sampling shape.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        final_shape = self.batch_shape + self.event_shape
        if shape == ():
            n = 1
        elif isinstance(shape, int):
            n = shape
            final_shape = (n,) + final_shape
        else:
            n = 1
            for shape_ in shape:
                n *= shape_
            final_shape = shape + final_shape
        samples = self._sample_n(n)
        return samples.reshape(final_shape)

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """Returns standard deviation.

        Returns:
            ~chainer.Variable: Output variable representing standard deviation.

        """
        raise NotImplementedError

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        raise NotImplementedError

    def survival_function(self, x):
        """Returns survival function for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing survival function
            for a input variable.

        """
        raise NotImplementedError

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        raise NotImplementedError
