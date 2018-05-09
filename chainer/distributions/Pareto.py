import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import exponential
import numpy


class Pareto(Distribution):

    """Pareto Distribution.

    Args:
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        alpha(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, scale, alpha):
        if isinstance(scale, chainer.Variable):
            self.scale = scale
        else:
            self.scale = chainer.Variable(scale)
        if isinstance(alpha, chainer.Variable):
            self.alpha = alpha
        else:
            self.alpha = chainer.Variable(alpha)

    def __copy__(self):
        return self._copy_to(Pareto(self.scale, self.a))

    @property
    def batch_shape(self):
        return self.scale.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return - exponential.log(self.alpha) + exponential.log(self.scale) \
            + 1. / self.alpha + 1.

    @property
    def event_shape(self):
        return ()

    @property
    def _is_gpu(self):
        return isinstance(self.scale.data, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        ba = broadcast.broadcast_to(self.alpha, x.shape)
        bs = broadcast.broadcast_to(self.scale, x.shape)
        if self._is_gpu:
            valid = cuda.cupy.zeros_like(x)
            inf = cuda.cupy.zeros_like(x)
        else:
            valid = numpy.zeros_like(x)
            inf = numpy.zeros_like(x)
        valid[x >= self.scale.data] = 1
        inf[x < self.scale.data] = numpy.inf
        return (exponential.log(ba)
                + ba * exponential.log(bs)
                - (ba + 1) * exponential.log(x)) * valid - inf

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        if self._is_gpu:
            valid = cuda.cupy.zeros_like(self.scale)
            inf = cuda.cupy.zeros_like(self.scale)
        else:
            valid = numpy.zeros_like(self.scale)
            inf = numpy.zeros_like(self.scale)
        valid[self.alpha.data > 1] = 1
        inf[self.alpha.data <= 1] = numpy.inf
        return (self.alpha * self.scale / (self.alpha - 1)) * valid + inf

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.pareto(
                cuda.to_cpu(self.alpha.data),
                (n,)+self.scale.shape).astype(numpy.float32)
            eps = cuda.to_gpu(
                eps, cuda.get_device_from_array(self.alpha.data).id)
        else:
            eps = numpy.random.pareto(
                self.alpha.data, (n,)+self.scale.shape).astype(numpy.float32)

        noise = repeat.repeat(
            expand_dims.expand_dims(self.scale, axis=0), n, axis=0) * (eps + 1)
        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return '[scale, inf]'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        if self._is_gpu:
            valid = cuda.cupy.zeros_like(self.scale)
            inf = cuda.cupy.zeros_like(self.scale)
        else:
            valid = numpy.zeros_like(self.scale)
            inf = numpy.zeros_like(self.scale)

        valid[self.alpha.data > 2] = 1
        inf[self.alpha.data <= 2] = numpy.inf
        return (self.scale ** 2 * self.alpha / (self.alpha - 1) ** 2
                / (self.alpha - 2)) * valid + inf
