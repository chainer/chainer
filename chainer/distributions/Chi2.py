import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class Chi2(Distribution):

    """Chi2 Distribution.

    Args:
        k(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, k):
        super(Chi2, self).__init__()
        if isinstance(k, chainer.Variable):
            self.k = k
        else:
            self.k = chainer.Variable(k)

    def __copy__(self):
        return self._copy_to(Chi2(self.k))

    @property
    def batch_shape(self):
        return self.k.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return 0.5 * self.k + numpy.log(2.) + lgamma.lgamma(0.5 * self.k) \
            + (1 - 0.5 * self.k) * digamma.digamma(0.5 * self.k)

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
        bk = broadcast.broadcast_to(self.k, x.shape)
        return - lgamma.lgamma(0.5 * bk) - 0.5 * bk * numpy.log(2.) \
            + (0.5 * bk - 1) * exponential.log(x) - 0.5 * x

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.k

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.chisquare(
                cuda.to_cpu(self.k.data),
                size=(n,)+self.k.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.k.data).id)
        else:
            eps = numpy.random.chisquare(
                self.k.data, size=(n,)+self.k.shape).astype(numpy.float32)

        noise = chainer.Variable(eps)
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
        return 2 * self.k
