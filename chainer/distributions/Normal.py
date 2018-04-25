from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import erf
from chainer.functions.math import erfinv
from chainer.functions.math import exponential
from chainer.functions.math import sum
import numpy


class Normal(Distribution):
    """Normal Distribution.

    """

    def __init__(self, loc, scale):
        """Initialize.

        Args:
            loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Parameter of distribution representing the
            location :math:`\\mu`.
            scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Parameter of distribution representing the
            scale :math:`\\log(\\sigma^2)`.

        """
        self.loc, self.scale = loc, scale

    def __copy__(self):
        return self._copy_to(Normal(self.loc, self.scale))

    @property
    def batch_shape(self):
        return self.loc.shape

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
        return 0.5 * (1. + erf.erf((x - self.loc) / (2 ** 0.5 * self.scale)))

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return sum.sum(exponential.log(self.scale)
                       + 0.5 * numpy.log(2 * numpy.pi * numpy.e))

    @property
    def event_shape(self):
        return ()

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
        return erfinv.erfinv(2 * x - 1) * (2 ** 0.5) * self.scale + self.mean

    @property
    def _is_gpu(self):
        return isinstance(self.loc, cuda.ndarray)

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
        return exponential.log(self.cdf(x))

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        return - 0.5 * numpy.log(2 * numpy.pi) - exponential.log(self.scale) \
               - 0.5 * (x - self.loc) ** 2 / self.scale ** 2

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
        return exponential.log(self.survival_function(x))

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.loc

    @property
    def mode(self):
        """Returns mode.

        Returns:
            ~chainer.Variable: Output variable representing mode.

        """
        return self.loc

    def prob(self, x):
        """Returns probability for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing probability.

        """
        return 1. / (2 * numpy.pi) ** 0.5 / self.scale * \
            exponential.exp(- 0.5 * (x - self.loc) ** 2 / self.scale ** 2)

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = cuda.cupy.random.standard_normal(
                (n,)+self.loc.shape, dtype=self.loc.dtype)
        else:
            eps = numpy.random.standard_normal(
                (n,)+self.loc.shape).astype(numpy.float32)

        noise = repeat.repeat(
            expand_dims.expand_dims(self.scale, axis=0), n, axis=0) * eps
        noise += repeat.repeat(expand_dims.expand_dims(
            self.loc, axis=0), n, axis=0)

        return noise

    @property
    def stddev(self):
        """Returns standard deviation.

        Returns:
            ~chainer.Variable: Output variable representing standard deviation.

        """
        return self.scale

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return 'real'

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
        return 0.5 * (1. - erf.erf((x - self.loc) / (2 ** 0.5 * self.scale)))

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return self.scale ** 2
