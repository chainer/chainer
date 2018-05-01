import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import exponential
import numpy


class LogNormal(Distribution):

    """LogNormal Distribution.

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, loc, scale):
        if isinstance(loc, chainer.Variable):
            self.loc = loc
        else:
            self.loc = chainer.Variable(loc)
        if isinstance(scale, chainer.Variable):
            self.scale = scale
        else:
            self.scale = chainer.Variable(scale)

    def __copy__(self):
        return self._copy_to(LogNormal(self.loc, self.scale))

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return 0.5 + 0.5 * numpy.log(2 * numpy.pi) \
            + exponential.log(self.scale) + self.loc

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
        return - 0.5 * numpy.log(2 * numpy.pi) - exponential.log(self.scale) \
            - exponential.log(x) \
            - (0.5 * (exponential.log(x) - self.loc) ** 2 / self.scale ** 2)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return exponential.exp(self.loc + 0.5 * self.scale ** 2)

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = cuda.cupy.random.standard_normal(
                (n,)+self.loc.shape, dtype=self.loc.dtype)
        else:
            eps = numpy.random.standard_normal(
                (n,)+self.loc.shape).astype(numpy.float32)

        noise = repeat.repeat(
            expand_dims.expand_dims(self.scale, axis=0), n, axis=0) * eps
        noise += repeat.repeat(expand_dims.expand_dims(
            self.loc, axis=0), n, axis=0)

        return exponential.exp(noise)

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
        return exponential.exp(2 * self.loc + self.scale ** 2) \
            * (exponential.exp(self.scale ** 2) - 1)
