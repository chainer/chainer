import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import ceil
from chainer.functions.math import exponential
import numpy


class Geometric(Distribution):

    """Geometric Distribution.

    Args:
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, p):
        if isinstance(p, chainer.Variable):
            self.p = p
        else:
            self.p = chainer.Variable(p)

    def __copy__(self):
        return self._copy_to(Geometric(self.p))

    @property
    def batch_shape(self):
        return self.p.shape

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.p, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        return (x - 1) * exponential.log(1 - self.p) + exponential.log(self.p)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return 1 / self.p

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.uniform(
                0, 1, size=(n,)+self.p.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.p).id)
        else:
            eps = numpy.random.uniform(
                0, 1, size=(n,)+self.p.shape).astype(numpy.float32)

        noise = ceil.ceil(exponential.log(1 - eps) / exponential.log(
            1 - repeat.repeat(expand_dims.expand_dims(self.p, axis=0),
                              n, axis=0)))
        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return 'positive integer'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return (1 - self.p) / self.p ** 2
