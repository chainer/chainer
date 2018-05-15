import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class Beta(Distribution):

    """Beta Distribution.

    Args:
        a(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        b(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, a, b):
        super(Beta, self).__init__()
        if isinstance(a, chainer.Variable):
            self.a = a
        else:
            self.a = chainer.Variable(a)
        if isinstance(b, chainer.Variable):
            self.b = b
        else:
            self.b = chainer.Variable(b)

    def __copy__(self):
        return self._copy_to(Beta(self.a, self.b))

    @property
    def batch_shape(self):
        return self.a.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return lgamma.lgamma(self.a) + lgamma.lgamma(self.b) \
            - lgamma.lgamma(self.a + self.b) \
            - (self.a - 1) * digamma.digamma(self.a) \
            - (self.b - 1) * digamma.digamma(self.b) \
            + (self.a + self.b - 2) * digamma.digamma(self.a + self.b)

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.a.data, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        ba = broadcast.broadcast_to(self.a, x.shape)
        bb = broadcast.broadcast_to(self.b, x.shape)
        if self._is_gpu:
            inf = cuda.cupy.zeros_like(ba.data)
            constraint = cuda.cupy.bitwise_and(
                x.data >= 0, x.data < 1)
            not_constraint = cuda.cupy.logical_not(constraint)
        else:
            inf = numpy.zeros_like(ba.data)
            constraint = numpy.bitwise_and(
                x.data >= 0, x.data < 1)
            not_constraint = numpy.logical_not(constraint)
        inf[not_constraint] = numpy.inf

        logp = (ba - 1) * exponential.log(x * constraint + not_constraint) \
            + (bb - 1) * exponential.log(
                (1 - x) * constraint + not_constraint) \
            - lgamma.lgamma(ba) - lgamma.lgamma(bb) \
            + lgamma.lgamma(ba + bb)
        return logp - inf

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.a / (self.a + self.b)

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.beta(
                cuda.to_cpu(self.a.data), cuda.to_cpu(self.b.data),
                size=(n,)+self.a.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.a.data).id)
        else:
            eps = numpy.random.beta(
                self.a.data, self.b.data,
                size=(n,)+self.a.shape).astype(numpy.float32)

        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return '[0, 1]'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return (self.a * self.b) / (self.a + self.b) ** 2 \
            / (self.a + self.b + 1)
