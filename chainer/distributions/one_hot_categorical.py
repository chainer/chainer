import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
import chainer.functions.math.sum as sum_mod


class OneHotCategorical(distribution.Distribution):

    """OneHotCategorical Distribution.

    Args:
        p(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution.
    """

    def __init__(self, p):
        super(OneHotCategorical, self).__init__()
        self.__p = chainer.as_variable(p)

    @property
    def p(self):
        return self.__p

    @property
    def batch_shape(self):
        return self.p.shape[:-1]

    @property
    def event_shape(self):
        return self.p.shape[-1:]

    @property
    def _is_gpu(self):
        return isinstance(self.p.data, cuda.ndarray)

    def log_prob(self, x):
        return sum_mod.sum(exponential.log(self.p) * x, axis=-1)

    @property
    def mean(self):
        return self.p

    def sample_n(self, n):
        xp = cuda.get_array_module(self.p)
        obo_p = self.p.data.reshape((-1,) + self.event_shape)
        eye = xp.eye(self.event_shape[0])
        eps = [xp.random.choice(
            one_p.shape[0], size=(n,), p=one_p) for one_p in obo_p]
        eps = xp.stack(eps).T.reshape((n,)+self.batch_shape)
        eps = eye[eps]
        noise = chainer.Variable(eps)
        return noise

    @property
    def variance(self):
        return self.p * (1. - self.p)
