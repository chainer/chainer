import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.math import exponential
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
        return isinstance(self.p.data, cuda.ndarray)

    def log_prob(self, x):
        """Returns logarithm logarithm of probability for a input variable.

        Args:
            x: Input variable representing a random variable.

        Returns:
            Output variable representing logarithm of probability.

        """
        mg = numpy.meshgrid(
            *tuple(range(i) for i in self.batch_shape), indexing='ij')
        if isinstance(x, chainer.Variable):
            return exponential.log(self.p)[mg + [x.data.astype(numpy.int32)]]
        else:
            return exponential.log(self.p)[mg + [x.astype(numpy.int32)]]

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
            eps = [cuda.cupy.random.choice(
                one_p.shape[0], size=(n,), p=one_p) for one_p in obo_p]
            eps = cuda.cupy.stack(eps).T.reshape((n,)+self.batch_shape)
        else:
            eps = [numpy.random.choice(
                one_p.shape[0], size=(n,), p=one_p) for one_p in obo_p]
            eps = numpy.stack(eps).T.reshape((n,)+self.batch_shape)
        noise = chainer.Variable(eps)
        return noise
