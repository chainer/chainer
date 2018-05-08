import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import exponential
from chainer.functions.math import trigonometric
import numpy


class Cauchy(Distribution):

    """Cauchy Distribution.

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\log(\\sigma^2)`.

    """

    def __init__(self, loc, scale):
        super(Cauchy, self).__init__()
        if isinstance(loc, chainer.Variable):
            self.loc = loc
        else:
            self.loc = chainer.Variable(loc)
        if isinstance(scale, chainer.Variable):
            self.scale = scale
        else:
            self.scale = chainer.Variable(scale)

    def __copy__(self):
        return self._copy_to(Cauchy(self.loc, self.scale))

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
        return 1 / numpy.pi * trigonometric.arctan(
            (x - broadcast.broadcast_to(self.loc, x.shape))
            / broadcast.broadcast_to(self.scale, x.shape)) + 0.5

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return exponential.log(4 * numpy.pi * self.scale)

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
        return broadcast.broadcast_to(self.loc, x.shape) \
            + broadcast.broadcast_to(self.scale, x.shape) \
            * trigonometric.tan((x - 0.5) * numpy.pi)

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
        bs = broadcast.broadcast_to(self.scale, x.shape)
        bl = broadcast.broadcast_to(self.loc, x.shape)
        return - numpy.log(numpy.pi) + exponential.log(bs) \
            - exponential.log((x - bl)**2 + bs**2)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.loc * numpy.nan

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.standard_cauchy(
                (n,)+self.loc.shape).astype(numpy.float32)
            eps = cuda.to_gpu(
                eps, cuda.get_device_from_array(self.loc.data).id)
        else:
            eps = numpy.random.standard_cauchy(
                (n,)+self.loc.shape).astype(numpy.float32)

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
        return self.loc * numpy.nan
