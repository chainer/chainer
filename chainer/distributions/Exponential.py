import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import exponential
import numpy


class Exponential(Distribution):

    """Exponential Distribution.

    Args:
        lam(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, lam):
        if isinstance(lam, chainer.Variable):
            self.lam = lam
        else:
            self.lam = chainer.Variable(lam)

    def __copy__(self):
        return self._copy_to(Exponential(self.lam))

    @property
    def batch_shape(self):
        return self.lam.shape

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
        return 1 - exponential.exp(
            -broadcast.broadcast_to(self.lam, x.shape) * x)

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return 1 - exponential.log(self.lam)

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
        return -1 / self.lam * exponential.log(1 - x)

    @property
    def _is_gpu(self):
        return isinstance(self.lam.data, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        bl = broadcast.broadcast_to(self.lam, x.shape)
        if self._is_gpu:
            inf = cuda.cupy.zeros(x.shape)
            constraint = x.data >= 0
            not_constraint = cuda.cupy.logical_not(constraint)
        else:
            inf = numpy.zeros(x.shape)
            constraint = x.data >= 0
            not_constraint = numpy.logical_not(constraint)
        inf[not_constraint] = numpy.inf

        return exponential.log(bl) - bl * x - inf

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return 1 / self.lam

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.exponential(
                size=(n,)+self.lam.shape).astype(numpy.float32)
            eps = cuda.to_gpu(
                eps, cuda.get_device_from_array(self.lam.data).id)
        else:
            eps = numpy.random.exponential(
                size=(n,)+self.lam.shape).astype(numpy.float32)

        noise = eps / broadcast.broadcast_to(self.lam, eps.shape)
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
        return 1 / self.lam ** 2
