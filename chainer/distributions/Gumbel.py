import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import exponential
import numpy


class Gumbel(Distribution):

    """FisherSnedecor Distribution.

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, loc, scale):
        self.euler = 0.57721566490153286060651209008240243104215933593992
        if isinstance(loc, chainer.Variable):
            self.loc = loc
        else:
            self.loc = chainer.Variable(loc)
        if isinstance(scale, chainer.Variable):
            self.scale = scale
        else:
            self.scale = chainer.Variable(scale)

    def __copy__(self):
        return self._copy_to(Gumbel(self.loc, self.scale))

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return exponential.log(self.scale) + self.euler + 1.

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.loc, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        y = (x - self.loc) / self.scale
        return - exponential.log(self.scale) - y - exponential.exp(-y)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.loc + self.euler * self.scale

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.gumbel(
                size=(n,)+self.loc.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.k).id)
        else:
            eps = numpy.random.gumbel(
                size=(n,)+self.loc.shape).astype(numpy.float32)

        noise = repeat.repeat(
            expand_dims.expand_dims(self.scale, axis=0), n, axis=0) * eps
        noise += repeat.repeat(expand_dims.expand_dims(
            self.loc, axis=0), n, axis=0)

        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return 'real'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return numpy.pi ** 2 * self.scale ** 2 / 6
