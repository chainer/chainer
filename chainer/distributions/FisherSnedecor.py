import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class FisherSnedecor(Distribution):

    """FisherSnedecor Distribution.

    Args:
        d1(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        d2(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, d1, d2):
        if isinstance(d1, chainer.Variable):
            self.d1 = d1
        else:
            self.d1 = chainer.Variable(d1)
        if isinstance(d2, chainer.Variable):
            self.d2 = d2
        else:
            self.d2 = chainer.Variable(d2)

    def __copy__(self):
        return self._copy_to(FisherSnedecor(self.d1, self.d2))

    @property
    def batch_shape(self):
        return self.d1.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.d1, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        return 0.5 * self.d1 * exponential.log(self.d1 * x) \
            + 0.5 * self.d2 * exponential.log(self.d2) \
            - 0.5 * (self.d1 + self.d2) \
            * exponential.log(self.d1 * x + self.d2) - exponential.log(x) \
            - lgamma.lgamma(0.5 * self.d1) - lgamma.lgamma(0.5 * self.d2) \
            + lgamma.lgamma(0.5 * (self.d1 + self.d2))

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        if self._is_gpu:
            valid = cuda.cupy.zeros_like(self.d2)
            inf = cuda.cupy.zeros_like(self.d2)
        else:
            valid = numpy.zeros_like(self.d2)
            inf = numpy.zeros_like(self.d2)
        valid[self.d2.data > 2] = 1
        inf[self.d2.data <= 2] = numpy.inf
        return (self.d2 / (self.d2 - 2)) * valid + inf

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.f(
                cuda.to_cpu(self.d1.data),
                cuda.to_cpu(self.d2.data),
                size=(n,)+self.d1.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.k).id)
        else:
            eps = numpy.random.f(
                self.d1.data, self.d2.data,
                size=(n,)+self.d1.shape).astype(numpy.float32)

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
        if self._is_gpu:
            valid = cuda.cupy.zeros_like(self.d2)
            inf = cuda.cupy.zeros_like(self.d2)
        else:
            valid = numpy.zeros_like(self.d2)
            inf = numpy.zeros_like(self.d2)
        valid[self.d2.data > 4] = 1
        inf[self.d2.data <= 4] = numpy.inf
        return (2 * self.d2 ** 2 * (self.d1 + self.d2 - 2)
                / self.d1 / (self.d2 - 2) ** 2 / (self.d2 - 4)) * valid + inf
