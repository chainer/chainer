import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class Poisson(Distribution):

    """Poisson Distribution.

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
        return self._copy_to(Poisson(self.lam))

    @property
    def batch_shape(self):
        return self.lam.shape

    @property
    def event_shape(self):
        return ()

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
        if isinstance(x, chainer.Variable):
            x32 = x.data.astype(numpy.float32)
        else:
            x32 = x.astype(numpy.float32)
        bl = broadcast.broadcast_to(self.lam, x.shape)
        return x32 * exponential.log(bl) - lgamma.lgamma(x32 + 1) \
            - bl

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.lam

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.poisson(
                cuda.to_cpu(self.lam.data),
                size=(n,)+self.lam.shape).astype(numpy.float32)
            eps = cuda.to_gpu(
                eps, cuda.get_device_from_array(self.lam.data).id)
        else:
            eps = numpy.random.poisson(
                self.lam.data, size=(n,)+self.lam.shape).astype(numpy.float32)

        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return 'non negative integer'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return self.lam
