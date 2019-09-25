import copy

from chainer import backend


class Distribution(object):

    """Interface of Distribution

    :class:`Distribution` is a bass class for dealing with probability
    distributions.

    This class provides the following capabilities.

    1. Sampling random points.

    2. Evaluating a probability-related function at a given realization \
    value. (e.g., probability density function, probability mass function)

    3. Obtaining properties of distributions. (e.g., mean, variance)

    Note that every method and property that computes them from
    :class:`chainer.Variable` can basically be differentiated.

    In this class, sampled random points and realization values given in
    probability-related function is called *sample*.  Sample consists of
    *batches*, and each batch consists of independent *events*. Each event
    consists of values, and each value in an event cannot be sampled
    independently in general. Each event in a batch is independent while it is
    not sampled from an identical distribution. And each batch in sample is
    sampled from an identical distribution.

    Each part of the sample-batch-event hierarchy has its own shape, which is
    called ``sample_shape``, ``batch_shape``, and ``event_shape``,
    respectively.

    On initialization, it takes distribution-specific parameters as inputs.
    :attr:`batch_shape` and :attr:`event_shape` is decided by the shape of
    the parameter when generating an instance of a class.

    .. admonition:: Example

        The following code is an example of sample-batch-event hierarchy on
        using :class:`~distributions.MultivariateNormal` distribution. This
        makes 2d normal distributions. ``dist`` consists of 12(4 * 3)
        independent 2d normal distributions. And on initialization,
        :attr:`batch_shape` and :attr:`event_shape` is decided.

        >>> import chainer
        >>> import chainer.distributions as D
        >>> import numpy as np
        >>> d = 2
        >>> shape = (4, 3)
        >>> loc = np.random.normal(
        ...     size=shape + (d,)).astype(np.float32)
        >>> cov = np.random.normal(size=shape + (d, d)).astype(np.float32)
        >>> cov = np.matmul(cov, np.rollaxis(cov, -1, -2))
        >>> l = np.linalg.cholesky(cov)
        >>> dist = D.MultivariateNormal(loc, scale_tril=l)
        >>> dist.event_shape
        (2,)
        >>> dist.batch_shape
        (4, 3)
        >>> sample = dist.sample(sample_shape=(6, 5))
        >>> sample.shape
        (6, 5, 4, 3, 2)

    Every probability-related function takes realization value whose shape is
    the concatenation of ``sample_shape``, ``batch_shape``, and
    ``event_shape`` and returns an evaluated value whose shape is the
    concatenation of ``sample_shape``, and ``batch_shape``.

    """

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        return target

    @property
    def batch_shape(self):
        """Returns the shape of a batch.

        Returns:
            tuple: The shape of a sample that is not identical and independent.

        """
        raise NotImplementedError

    def cdf(self, x):
        """Evaluates the cumulative distribution function at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

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
        """Returns the shape of an event.

        Returns:
            tuple: The shape of a sample that is not identical and independent.

        """
        raise NotImplementedError

    def icdf(self, x):
        """Evaluates the inverse cumulative distribution function at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

        Returns:
            ~chainer.Variable: Inverse cumulative distribution function value
            evaluated at `x`.

        """
        raise NotImplementedError

    def log_cdf(self, x):
        """Evaluates the log of cumulative distribution function at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

        Returns:
            ~chainer.Variable: Logarithm of cumulative distribution function
            value evaluated at `x`.

        """
        raise NotImplementedError

    def log_prob(self, x):
        """Evaluates the logarithm of probability at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

        Returns:
            ~chainer.Variable: Logarithm of probability evaluated at `x`.

        """
        raise NotImplementedError

    def log_survival_function(self, x):
        """Evaluates the logarithm of survival function at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

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

    @property
    def params(self):
        """Returns the parameters of the distribution.

        Returns:
            dict: The parameters of the distribution.

        """
        raise NotImplementedError

    def perplexity(self, x):
        """Evaluates the perplexity function at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

        Returns:
            ~chainer.Variable: Perplexity function value evaluated at `x`.

        """
        raise NotImplementedError

    def prob(self, x):
        """Evaluates probability at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

        Returns:
            ~chainer.Variable: Probability evaluated at `x`.

        """
        raise NotImplementedError

    def sample(self, sample_shape=()):
        """Samples random points from the distribution.

        This function calls `sample_n` and reshapes a result of `sample_n` to
        `sample_shape + batch_shape + event_shape`. On implementing sampling
        code in an inherited distribution class, it is not recommended that
        you override this function. Instead of doing this, it is preferable to
        override `sample_n`.

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
        samples = self.sample_n(n)
        return samples.reshape(final_shape)

    def sample_n(self, n):
        """Samples n random points from the distribution.

        This function returns sampled points whose shape is
        `(n,) + batch_shape + event_shape`. When implementing sampling code in
        a subclass, it is recommended that you override this method.

        Args:
            n(int): Sampling size.

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
            str: String that means support of this distribution.

        """
        raise NotImplementedError

    def survival_function(self, x):
        """Evaluates the survival function at the given points.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the domain of the distribution

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

    @property
    def xp(self):
        """Array module for the distribution.

        Depending on which of CPU/GPU this distribution is on, this property
        returns :mod:`numpy` or :mod:`cupy`.
        """
        return backend.get_array_module(*self.params.values())


_KLDIVERGENCE = {}


def register_kl(Dist1, Dist2):
    """Decorator to register KL divergence function.

    This decorator registers a function which computes Kullback-Leibler
    divergence. This function will be called by :func:`~chainer.kl_divergence`
    based on the argument types.

    Args:
        Dist1(`type`): type of a class inherit from
            :class:`~chainer.Distribution` to calculate KL divergence.
        Dist2(`type`): type of a class inherit from
            :class:`~chainer.Distribution` to calculate KL divergence.

    The decorated functoion takes an instance of ``Dist1`` and ``Dist2`` and
    returns KL divergence value.

    .. admonition:: Example

        This is a simple example to register KL divergence. A function to
        calculate a KL divergence value between an instance of ``Dist1`` and
        an instance of ``Dist2`` is registered.

        .. code-block:: python

            from chainer import distributions
            @distributions.register_kl(Dist1, Dist2)
            def _kl_dist1_dist2(dist1, dist2):
                return KL

    """
    def f(kl):
        _KLDIVERGENCE[Dist1, Dist2] = kl
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

    Using :func:`~chainer.register_kl`, we can define behavior of
    :func:`~chainer.kl_divergence` for any two distributions.

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

    This function call :func:`~chainer.kl_divergence` and
    :meth:`~chainer.Distribution.entropy` of ``dist1``. Therefore, it is
    necessary to register KL divergence function with
    :func:`~chainer.register_kl` decoartor and define
    :meth:`~chainer.Distribution.entropy` in ``dist1``.

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
