import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.math import basic_math
from chainer.functions.math import clip
from chainer.functions.math import exponential
from chainer.functions.math import sign
from chainer.functions.math import sqrt
import numpy


class Laplace(Distribution):

    """Laplace Distribution.

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\log(\\sigma^2)`.

    """

    def __init__(self, loc, scale):
        super(Laplace, self).__init__()
        if isinstance(loc, chainer.Variable):
            self.loc = loc
        else:
            self.loc = chainer.Variable(loc)
        if isinstance(scale, chainer.Variable):
            self.scale = scale
        else:
            self.scale = chainer.Variable(scale)

    def __copy__(self):
        return self._copy_to(Laplace(self.loc, self.scale))

    @property
    def batch_shape(self):
        return self.loc.shape

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
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return clip.clip(0.5 * exponential.exp(
            (x - bl) / bs), 0., 0.5) \
            + clip.clip(0.5 - 0.5 * exponential.exp(
                -(x - bl) / bs), 0., 0.5)

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return 1. + exponential.log(2 * self.scale)

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
        return self.loc - self.scale * sign.sign(x - 0.5) \
            * exponential.log(- basic_math.absolute(2 * x - 1) + 1)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.data, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return - exponential.log(2 * bs) \
            - basic_math.absolute(x - bl) / bs

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.loc

    @property
    def mode(self):
        """Returns mode.

        Returns:
            ~chainer.Variable: Output variable representing mode.

        """
        return self.loc

    def prob(self, x):
        """Returns probability for a input variable.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable representing a random
            variable.

        Returns:
            ~chainer.Variable: Output variable representing probability.

        """
        bl = broadcast.broadcast_to(self.loc, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        return 0.5 / bs * exponential.exp(- basic_math.absolute(x - bl) / bs)

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.laplace(
                size=(n,) + self.loc.shape).astype(numpy.float32)
            eps = cuda.to_gpu(
                eps, cuda.get_device_from_array(self.loc.data).id)
        else:
            eps = numpy.random.laplace(
                size=(n,) + self.loc.shape).astype(numpy.float32)

        noise = broadcast.broadcast_to(self.scale, eps.shape) * eps
        noise += broadcast.broadcast_to(self.loc, eps.shape)

        return noise

    @property
    def stddev(self):
        """Returns standard deviation.

        Returns:
            ~chainer.Variable: Output variable representing standard deviation.

        """
        return sqrt.sqrt(2 * self.scale ** 2)

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
        return 2 * self.scale ** 2
