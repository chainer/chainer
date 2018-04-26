from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import stack
from chainer.functions.math import exponential
from chainer.functions.math import sum
from chainer import Variable
import numpy


class Bernoulli(Distribution):

    """Bernoulli Distribution.

    Args:
        p (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing \
        probability that variable is 1.

    """

    def __init__(self, p):
        self.p = p

    def __copy__(self):
        return self._copy_to(Bernoulli(self.p))

    @property
    def batch_shape(self):
        """Returns the shape of a sample.

        Returns:
            ~chainer.Variable: Output variable representing the shape of a
            sample.

        """
        return self.p.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            ~chainer.Variable: Output variable representing entropy.

        """
        return -self.p * exponential.log(self.p) - \
            (1 - self.p) * exponential.log(1 - self.p)

    @property
    def enumerate_support(self):
        """Returns support values of discrete distribution.

        Returns:
            ~chainer.Variable: Output variable containing candidates.

        """
        if self._is_gpu:
            zeros = cuda.cupy.zeros_like(self.p)
            ones = cuda.cupy.ones_like(self.p)
        else:
            zeros = numpy.zeros_like(self.p)
            ones = numpy.ones_like(self.p)
        return stack.stack([zeros, ones], axis=-1)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.p, cuda.ndarray)

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
        return x * exponential.log(self.p) \
            + (1 - x) * exponential.log(1 - self.p)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.p

    @property
    def mode(self):
        """Returns mode.

        Returns:
            ~chainer.Variable: Output variable representing mode.

        """
        if isinstance(self.p, Variable):
            return Variable((self.p.data > 0.5) * 1.0)
        else:
            return Variable((self.p > 0.5) * 1.0)

    def prob(self, x):
        """Returns probability for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing probability.

        """
        return x * self.p + (1 - x) * (1 - self.p)

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            rand = cuda.cupy.random.uniform(size=(n,)+self.p.shape)
        else:
            rand = numpy.random.uniform(size=(n,)+self.p.shape)

        if isinstance(self.p, Variable):
            criteria = self.p.data
        else:
            criteria = self.p
        criteria = numpy.repeat(numpy.expand_dims(criteria, axis=0), n, axis=0)
        return Variable((rand < criteria) * 1.0)

    @property
    def stddev(self):
        """Returns standard deviation.

        Returns:
            ~chainer.Variable: Output variable representing standard deviation.

        """
        return (self.p * (1 - self.p)) ** 0.5

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return (self.p * (1 - self.p))
