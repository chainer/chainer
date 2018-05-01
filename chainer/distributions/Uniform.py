import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import clip
from chainer.functions.math import exponential
import numpy


class Uniform(Distribution):

    """Uniform Distribution.

    Args:
        low(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`.
        high(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\log(\\sigma^2)`.

    """

    def __init__(self, low, high):
        if isinstance(low, chainer.Variable):
            self.low = low
        else:
            self.low = chainer.Variable(low)
        if isinstance(high, chainer.Variable):
            self.high = high
        else:
            self.high = chainer.Variable(high)

    def __copy__(self):
        return self._copy_to(Uniform(self.low, self.high))

    @property
    def batch_shape(self):
        return self.low.shape

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
        return clip.clip((x - self.low)/(self.high - self.low), 0., 1.)

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return exponential.log(self.high - self.low)

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
        return x * broadcast.broadcast_to(self.high, x.shape) \
            + (1 - x) * broadcast.broadcast_to(self.low, x.shape)

    @property
    def _is_gpu(self):
        return isinstance(self.low, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        if self._is_gpu:
            logp = cuda.cupy.zeros_like(self.low)
            inf = cuda.cupy.zeros_like(self.low)
            constraint = cuda.cupy.bitwise_and(
                x.data >= self.low.data, x.data < self.high.data)
            not_constraint = cuda.cupy.logical_not(constraint)
        else:
            logp = numpy.zeros_like(self.low)
            inf = numpy.zeros_like(self.low)
            constraint = numpy.bitwise_and(
                x.data >= self.low.data, x.data < self.high.data)
            not_constraint = numpy.logical_not(constraint)
        logp[constraint] = 1
        logp *= -exponential.log(self.high - self.low)
        inf[not_constraint] = numpy.inf
        return logp - inf

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return (self.high + self.low) / 2

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = cuda.cupy.random.uniform(
                0, 1, (n,)+self.low.shape, dtype=self.low.dtype)
        else:
            eps = numpy.random.uniform(
                0, 1, (n,)+self.low.shape).astype(numpy.float32)

        noise = self.icdf(eps)

        return noise

    @property
    def stddev(self):
        """Returns standard deviation.

        Returns:
            ~chainer.Variable: Output variable representing standard deviation.

        """
        return exponential.log(self.variance)

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return "[low, high]"

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return (self.high - self.low) ** 2 / 12
