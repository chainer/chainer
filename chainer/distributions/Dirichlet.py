import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.array import expand_dims
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.functions.math import sum
import numpy


class Dirichlet(Distribution):

    """Dirichlet Distribution.

    Args:
        alpha(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, alpha):
        if isinstance(alpha, chainer.Variable):
            self.alpha = alpha
        else:
            self.alpha = chainer.Variable(alpha)
        self.k = self.alpha.shape[-1]

    @property
    def alpha0(self):
        return sum.sum(self.alpha, axis=-1)

    def __copy__(self):
        return self._copy_to(Dirichlet(self.alpha))

    @property
    def batch_shape(self):
        return self.alpha.shape[:-1]

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        return sum.sum(lgamma.lgamma(self.alpha), axis=-1) - lgamma.lgamma(self.alpha0) \
            + (self.alpha0 - self.k) * digamma.digamma(self.alpha0) \
            - sum.sum((self.alpha - 1) * digamma.digamma(self.alpha), axis=-1)

    @property
    def event_shape(self):
        return self.alpha.shape[-1:]

    @property
    def _is_gpu(self):
        return isinstance(self.alpha, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        return - sum.sum(lgamma.lgamma(self.alpha), axis=-1) \
            + lgamma.lgamma(self.alpha0) \
            + sum.sum((self.alpha - 1) * exponential.log(x), axis=-1)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        br_alpha0 = expand_dims.expand_dims(self.alpha0, axis=-1)
        br_alpha0 = broadcast.broadcast_to(br_alpha0, self.alpha.shape)
        return self.alpha / br_alpha0

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        if self._is_gpu:
            obo_alpha = self.alpha.data.reshape(-1, self.k)
            eps = [numpy.random.dirichlet(cuda.to_cpu(one_alpha),
                   size=(n,)).astype(numpy.float32)
                   for one_alpha in obo_alpha]
            eps = numpy.stack(eps).reshape((n,) + self.alpha.shape)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.alpha).id)
        else:
            obo_alpha = self.alpha.data.reshape(-1, self.k)
            eps = [numpy.random.dirichlet(one_alpha,
                   size=(n,)).astype(numpy.float32)
                   for one_alpha in obo_alpha]
            eps = numpy.stack(eps).reshape((n,) + self.alpha.shape)

        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        """Returns support.

        Returns:
            string: Output string that means support of this distribution.

        """
        return '[0,1]'

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        br_alpha0 = expand_dims.expand_dims(self.alpha0, axis=-1)
        br_alpha0 = broadcast.broadcast_to(br_alpha0, self.alpha.shape)
        return self.alpha * (br_alpha0 - self.alpha) \
            / br_alpha0 ** 2 / (br_alpha0 + 1)
