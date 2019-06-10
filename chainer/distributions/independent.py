import numpy

from chainer.backend import cuda
from chainer import distribution
from chainer.functions.array import repeat
from chainer.functions.array import reshape
from chainer.functions.array import transpose
from chainer.functions.math import prod
from chainer.functions.math import sum as sum_mod
from chainer.utils import array
from chainer.utils import cache


class Independent(distribution.Distribution):

    """Independent distribution.

    Args:
        distribution (:class:`~chainer.Distribution`): The base distribution
            instance to transform.
        reinterpreted_batch_ndims (:class:`int`): Integer number of rightmost
            batch dims which will be regarded as event dims. When ``None`` all
            but the first batch axis (batch axis 0) will be transferred to
            event dimensions.
    """

    def __init__(self, distribution, reinterpreted_batch_ndims=None):
        super(Independent, self).__init__()
        self.__distribution = distribution
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = \
                self._get_default_reinterpreted_batch_ndims(distribution)
        elif reinterpreted_batch_ndims > len(distribution.batch_shape):
            raise ValueError(
                'reinterpreted_batch_ndims must be less than or equal to the '
                'number of dimensions of `distribution.batch_shape`.')
        self.__reinterpreted_batch_ndims = reinterpreted_batch_ndims

        batch_ndim = \
            len(self.distribution.batch_shape) - self.reinterpreted_batch_ndims
        self.__batch_shape = distribution.batch_shape[:batch_ndim]
        self.__event_shape = \
            distribution.batch_shape[batch_ndim:] + distribution.event_shape

    @property
    def distribution(self):
        return self.__distribution

    @property
    def reinterpreted_batch_ndims(self):
        return self.__reinterpreted_batch_ndims

    @property
    def batch_shape(self):
        return self.__batch_shape

    @property
    def event_shape(self):
        return self.__event_shape

    @cache.cached_property
    def covariance(self):
        """ The covariance of the independent distribution.

        By definition, the covariance of the new
        distribution becomes block diagonal matrix. Let
        :math:`\\Sigma_{\\mathbf{x}}` be the covariance matrix of the original
        random variable :math:`\\mathbf{x} \\in \\mathbb{R}^d`, and
        :math:`\\mathbf{x}^{(1)}, \\mathbf{x}^{(2)}, \\cdots \\mathbf{x}^{(m)}`
        be the :math:`m` i.i.d. random variables, new covariance matrix
        :math:`\\Sigma_{\\mathbf{y}}` of :math:`\\mathbf{y} =
        [\\mathbf{x}^{(1)}, \\mathbf{x}^{(2)}, \\cdots, \\mathbf{x}^{(m)}] \\in
        \\mathbb{R}^{md}` can be written as

        .. math::
            \\left[\\begin{array}{ccc}
                    \\Sigma_{\\mathbf{x}^{1}} & & 0 \\\\
                    & \\ddots & \\\\
                    0 & & \\Sigma_{\\mathbf{x}^{m}}
            \\end{array} \\right].

        Note that this relationship holds only if the covariance matrix of the
        original distribution is given analytically.

        Returns:
            ~chainer.Variable: The covariance of the distribution.
        """
        num_repeat = array.size_of_shape(
            self.distribution.batch_shape[-self.reinterpreted_batch_ndims:])
        dim = array.size_of_shape(self.distribution.event_shape)
        cov = repeat.repeat(
            reshape.reshape(
                self.distribution.covariance,
                ((self.batch_shape) + (1, num_repeat, dim, dim))),
            num_repeat, axis=-4)
        cov = reshape.reshape(
            transpose.transpose(
                cov, axes=(
                    tuple(range(len(self.batch_shape))) + (-4, -2, -3, -1))),
            self.batch_shape + (num_repeat * dim, num_repeat * dim))
        block_indicator = self.xp.reshape(
            self._block_indicator,
            tuple([1] * len(self.batch_shape)) + self._block_indicator.shape)
        return cov * block_indicator

    @property
    def entropy(self):
        return self._reduce(sum_mod.sum, self.distribution.entropy)

    def cdf(self, x):
        return self._reduce(prod.prod, self.distribution.cdf(x))

    def icdf(self, x):
        """The inverse cumulative distribution function for multivariate variable.

        Cumulative distribution function for multivariate variable is not
        invertible. This function always raises :class:`RuntimeError`.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Data points in
                the codomain of the distribution

        Raises:
            :class:`RuntimeError`
        """

        raise RuntimeError(
            'Cumulative distribution function for multivariate variable '
            'is not invertible.')

    def log_cdf(self, x):
        return self._reduce(sum_mod.sum, self.distribution.log_cdf(x))

    def log_prob(self, x):
        return self._reduce(sum_mod.sum, self.distribution.log_prob(x))

    def log_survival_function(self, x):
        return self._reduce(
            sum_mod.sum, self.distribution.log_survival_function(x))

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def mode(self):
        return self.distribution.mode

    @property
    def params(self):
        return self.distribution.params

    def perplexity(self, x):
        return self._reduce(prod.prod, self.distribution.perplexity(x))

    def prob(self, x):
        return self._reduce(prod.prod, self.distribution.prob(x))

    def sample_n(self, n):
        return self.distribution.sample_n(n)

    @property
    def stddev(self):
        return self.distribution.stddev

    @property
    def support(self):
        return self.distribution.support

    def survival_function(self, x):
        return self._reduce(prod.prod, self.distribution.survival_function(x))

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def xp(self):
        return self.distribution.xp

    def _reduce(self, op, stat):
        range_ = tuple(range(-self.reinterpreted_batch_ndims, 0))
        return op(stat, axis=range_)

    def _get_default_reinterpreted_batch_ndims(self, distribution):
        ndims = len(distribution.batch_shape)
        return max(0, ndims - 1)

    @cache.cached_property
    def _block_indicator(self):
        num_repeat = array.size_of_shape(
            self.distribution.batch_shape[-self.reinterpreted_batch_ndims:])
        dim = array.size_of_shape(self.distribution.event_shape)
        block_indicator = numpy.fromfunction(
            lambda i, j: i // dim == j // dim,
            (num_repeat * dim, num_repeat * dim)).astype(int)
        if self.xp is cuda.cupy:
            block_indicator = cuda.to_gpu(block_indicator)
        return block_indicator


@distribution.register_kl(Independent, Independent)
def _kl_independent_independent(dist1, dist2):
    """Computes Kullback-Leibler divergence for independent distributions.

    We can leverage the fact that
    .. math::
        \\mathrm{KL}(
                \\mathrm{Independent}(\\mathrm{dist1}) ||
                \\mathrm{Independent}(\\mathrm{dist2}))
        = \\mathrm{sum}(\\mathrm{KL}(\\mathrm{dist1} || \\mathrm{dist2}))
    where the sum is over the ``reinterpreted_batch_ndims``.

    Args:
        dist1 (:class:`~chainer.distribution.Independent`): Instance of
            `Independent`.
        dist2 (:class:`~chainer.distribution.Independent`): Instance of
            `Independent`.

    Returns:
        Batchwise ``KL(dist1 || dist2)``.

    Raises:
        :class:`ValueError`: If the event space for ``dist1`` and ``dist2``,
            or their underlying distributions don't match.
    """

    p = dist1.distribution
    q = dist2.distribution

    # The KL between any two (non)-batched distributions is a scalar.
    # Given that the KL between two factored distributions is the sum, i.e.
    # KL(p1(x)p2(y) || q1(x)q2(y)) = KL(p1 || q1) + KL(q1 || q2), we compute
    # KL(p || q) and do a `reduce_sum` on the reinterpreted batch dimensions.
    if dist1.event_shape == dist2.event_shape:
        if p.event_shape == q.event_shape:
            num_reduce_dims = len(dist1.event_shape) - len(p.event_shape)
            reduce_dims = tuple([-i - 1 for i in range(0, num_reduce_dims)])

            return sum_mod.sum(
                distribution.kl_divergence(p, q), axis=reduce_dims)
        else:
            raise NotImplementedError(
                'KL between Independents with different '
                'event shapes not supported.')
    else:
        raise ValueError('Event shapes do not match.')
