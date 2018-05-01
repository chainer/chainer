import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
import numpy


class StudentT(Distribution):

    """StudentT Distribution.

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`.
        scale(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`\\log(\\sigma^2)`.

    """

    def __init__(self, nu, loc, scale):
        if isinstance(nu, chainer.Variable):
            self.nu = nu
        else:
            self.nu = chainer.Variable(nu)
        if isinstance(loc, chainer.Variable):
            self.loc = loc
        else:
            self.loc = chainer.Variable(loc)
        if isinstance(scale, chainer.Variable):
            self.scale = scale
        else:
            self.scale = chainer.Variable(scale)

    def __copy__(self):
        return self._copy_to(StudentT(self.nu, self.loc, self.scale))

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        lgamma05 = 0.57236494292469997
        return exponential.log(self.scale) \
            + 0.5 * (self.nu + 1) * (digamma.digamma(0.5 * (1 + self.nu))
                                     - digamma.digamma(0.5 * self.nu)) \
            + 0.5 * exponential.log(self.nu) \
            + lgamma.lgamma(0.5 * self.nu) + lgamma05 \
            - lgamma.lgamma(0.5 * self.nu + 0.5)

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
        return lgamma.lgamma(0.5 * (self.nu + 1)) \
            - 0.5 * exponential.log(self.nu * numpy.pi) \
            - lgamma.lgamma(0.5 * self.nu) \
            - (0.5 * (self.nu + 1)) * exponential.log(
                1 + ((x - self.loc)/self.scale) ** 2 / self.nu) \
            - exponential.log(self.scale)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.loc

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            eps = numpy.random.standard_t(
                df=self.nu.data,
                size=(n,)+self.loc.shape).astype(numpy.float32)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.loc).id)
        else:
            eps = numpy.random.standard_t(
                df=self.nu.data,
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
        if self._is_gpu:
            standard_var = cuda.cupy.zeros_like(self.nu)
            standard_var[self.nu.data > 2.] \
                = (self.nu / (self.nu - 2.))[self.nu.data > 2.].data
            standard_var[self.nu.data <= 2.] = numpy.inf
        else:
            standard_var = numpy.zeros_like(self.nu)
            standard_var[self.nu.data > 2.] \
                = (self.nu / (self.nu - 2.))[self.nu.data > 2.].data
            standard_var[self.nu.data <= 2.] = numpy.inf

        return self.scale ** 2 * standard_var
