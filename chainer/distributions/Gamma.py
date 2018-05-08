import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class Gamma(Distribution):

    """Gamma Distribution.

    Args:
        k(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        theta(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, k, theta):
        if isinstance(k, chainer.Variable):
            self.k = k
        else:
            self.k = chainer.Variable(k)
        if isinstance(theta, chainer.Variable):
            self.theta = theta
        else:
            self.theta = chainer.Variable(theta)

    def __copy__(self):
        return self._copy_to(Gamma(self.k, self.theta))

    @property
    def batch_shape(self):
        return self.k.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return self.k + exponential.log(self.theta) + lgamma.lgamma(self.k) \
            + (1 - self.k) * digamma.digamma(self.k)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.k.data, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        return - lgamma.lgamma(self.k) - self.k * exponential.log(self.theta) \
            + (self.k - 1) * exponential.log(x) - x / self.theta

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.k * self.theta

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.gamma(
                cuda.to_cpu(self.k.data),
                size=(n,)+self.k.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.k.data).id)
        else:
            eps = numpy.random.gamma(
                self.k.data, size=(n,)+self.k.shape).astype(numpy.float32)

        noise = broadcast.broadcast_to(self.theta, eps.shape) * eps

        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return 'positive'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return self.k * self.theta * self.theta
