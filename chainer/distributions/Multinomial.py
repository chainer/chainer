import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.functions.math import sum
import numpy


class Multinomial(Distribution):

    """Multinomial Distribution.

    Args:
        n(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, n, p):
        if isinstance(n, chainer.Variable):
            self.n = n
        else:
            self.n = chainer.Variable(n)
        if isinstance(p, chainer.Variable):
            self.p = p
        else:
            self.p = chainer.Variable(p)

    def __copy__(self):
        return self._copy_to(Multinomial(self.n, self.p))

    @property
    def batch_shape(self):
        return self.p.shape[:-1]

    @property
    def event_shape(self):
        return self.p.shape[-1:]

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
        n32 = self.n.data.astype(numpy.float32)
        if isinstance(x, chainer.Variable):
            x32 = x.data.astype(numpy.float32)
        else:
            x32 = x.astype(numpy.float32)
        return lgamma.lgamma(n32 + 1) \
            - sum.sum(lgamma.lgamma(x32 + 1), axis=-1) \
            + sum.sum(x32 * exponential.log(self.p), axis=-1)

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.n * self.p

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        obo_p = self.p.data.reshape(-1, self.p.shape[-1])
        obo_n = numpy.broadcast_to(self.n.data, self.batch_shape).reshape(-1)
        if self._is_gpu:
            eps = [numpy.random.multinomial(
                one_n, one_p, size=n) for one_n, one_p
                in zip(obo_n, obo_p)]
            eps = numpy.stack(eps)
            eps = numpy.swapaxes(eps, 0, 1)
            eps = eps.reshape((n,)+self.batch_shape+(self.p.shape[-1],))
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.p).id)
        else:
            eps = [numpy.random.multinomial(
                one_n, one_p, size=n) for one_n, one_p
                in zip(obo_n, obo_p)]
            eps = numpy.stack(eps)
            eps = numpy.swapaxes(eps, 0, 1)
            eps = eps.reshape((n,)+self.batch_shape+(self.p.shape[-1],))
        noise = chainer.Variable(eps)
        return noise

    @property
    def variance(self):
        """Returns variance.

        Returns:
            ~chainer.Variable: Output variable representing variance.

        """
        return self.n * self.p * (1. - self.p)
