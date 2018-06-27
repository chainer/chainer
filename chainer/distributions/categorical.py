import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
from chainer.functions.math import sum as sum_mod
import numpy


class Categorical(distribution.Distribution):

    """Categorical Distribution.

    Args:
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.

    """

    def __init__(self, p):
        self.p = chainer.as_variable(p)

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
        mg = numpy.meshgrid(
            *tuple(range(i) for i in self.batch_shape), indexing='ij')
        if isinstance(x, chainer.Variable):
            return exponential.log(self.p)[mg + [x.data.astype(numpy.int32)]]
        else:
            return exponential.log(self.p)[mg + [x.astype(numpy.int32)]]

    def sample_n(self, n):
        onebyone_p = self.p.data.reshape(-1, self.p.shape[-1])
        if self._is_gpu:
            eps = [cuda.cupy.random.choice(
                one_p.shape[0], size=(n,), p=one_p) for one_p in onebyone_p]
            eps = cuda.cupy.stack(eps).T.reshape((n,)+self.batch_shape)
        else:
            eps = [numpy.random.choice(
                one_p.shape[0], size=(n,), p=one_p) for one_p in onebyone_p]
            eps = numpy.stack(eps).T.reshape((n,)+self.batch_shape)
        noise = chainer.Variable(eps)
        return noise


@distribution.register_kl(Categorical, Categorical)
def _kl_categorical_categorical(dist1, dist2):
    return sum_mod.sum(dist1.p * (
        exponential.log(dist1.p) - exponential.log(dist2.p)), axis=-1)
