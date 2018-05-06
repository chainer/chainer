import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.math import exponential
from chainer.functions.math import sum
import numpy


class Categorical(Distribution):

    """Categorical Distribution.

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
        return self._copy_to(Categorical(self.p))

    @property
    def batch_shape(self):
        return self.p.shape[:-1]

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
        mg = numpy.meshgrid(
            *tuple(range(i) for i in self.batch_shape), indexing='ij')
        return sum.sum(exponential.log(self.p)[mg + [x.astype(numpy.int32)]])

    def _sample_n(self, n):
        """Samples from this distribution.

        Args:
            n(`int`): Sampling size.

        Returns:
            ~chainer.Variable: Output variable representing sampled random
            variable.
        """
        obo_p = self.p.data.reshape(-1, self.p.shape[-1])
        if self._is_gpu:
            eps = [numpy.random.choice(
                one_p.shape[0], size=(n,), p=one_p) for one_p in obo_p]
            eps = numpy.stack(eps).T.reshape((n,)+self.batch_shape)
            eps = cuda.to_gpu(eps, cuda.get_device_from_array(self.p).id)
        else:
            eps = [numpy.random.choice(
                one_p.shape[0], size=(n,), p=one_p) for one_p in obo_p]
            eps = numpy.stack(eps).T.reshape((n,)+self.batch_shape)
        noise = chainer.Variable(eps)
        return noise
