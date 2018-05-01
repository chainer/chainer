import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class Binomial(Distribution):

    """Binomial Distribution.

    Args:
        n(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, n, p):
        if isinstance(n, chainer.Variable):
            self.n = n
        else:
            self.n = chainer.Variable(n)
        if isinstance(p, chainer.Variable):
            self.p = p
        else:
            self.p = chainer.Variable(p)

    def __copy__(self):
        return self._copy_to(Binomial(self.n, self.p))

    @property
    def batch_shape(self):
        return self.n.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.n, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        n32 = self.n.data.astype(numpy.float32)
        if isinstance(x, chainer.Variable):
            x32 = x.data.astype(numpy.float32)
        else:
            x32 = x.astype(numpy.float32)
        return lgamma.lgamma(n32 + 1) - lgamma.lgamma(x32 + 1) \
            - lgamma.lgamma(n32 - x32 + 1) + x * exponential.log(self.p) \
            + (n32 - x32) * exponential.log(1 - self.p)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.n.data * self.p

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.binomial(cuda.to_cpu(self.n.data),
                                        cuda.to_cpu(self.p.data),
                                        size=(n,)+self.n.shape)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.n).id)
        else:
            eps = numpy.random.binomial(self.n.data, self.p.data,
                                        size=(n,)+self.n.shape)

        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return '[0, n]'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return self.n.data * self.p * (1 - self.p)
