import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import exponential
import chainer.functions.math.sum as sum_mod
from chainer.utils import cache


def _stack(xp, xs, axis):
    try:
        return xp.stack(xs, axis)
    except AttributeError:
        # in case numpy<1.10, which does not have numpy.stack
        return xp.concatenate(
            [xp.expand_dims(x, axis) for x in xs],
            axis=axis)


def _random_choice(xp, a, size, p):
    try:
        return xp.random.choice(a, size, p=p)
    except ValueError:
        # Validate the sum of the probabilities as NumPy PR #6131 (numpy>=1.10)
        tol = xp.finfo(p.dtype).eps ** 0.5
        p = p.astype(xp.float64)
        xp.testing.assert_allclose(p.sum(), 1, rtol=0, atol=tol)

        # Normalize the probabilities as they satisfy the validation above, and
        # generate samples again
        p /= p.sum()
        return xp.random.choice(a, size, p=p)


class OneHotCategorical(distribution.Distribution):

    """OneHotCategorical Distribution.

    Args:
        p(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
    """

    def __init__(self, p):
        super(OneHotCategorical, self).__init__()
        self.__p = p

    @cache.cached_property
    def p(self):
        return chainer.as_variable(self.__p)

    @cache.cached_property
    def log_p(self):
        return exponential.log(self.p)

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
        return sum_mod.sum(self.log_p * x, axis=-1)

    @cache.cached_property
    def mean(self):
        return self.p

    @property
    def params(self):
        return {'p': self.p}

    def sample_n(self, n):
        xp = chainer.backend.get_array_module(self.p)
        obo_p = self.p.data.reshape((-1,) + self.event_shape)
        eye = xp.eye(self.event_shape[0], dtype=self.p.dtype)
        eps = [_random_choice(xp, one_p.shape[0], size=(n,), p=one_p)
               for one_p in obo_p]
        eps = _stack(xp, eps, axis=1).reshape((n,)+self.batch_shape)
        eps = eye[eps]
        noise = chainer.Variable(eps)
        return noise

    @cache.cached_property
    def variance(self):
        return self.p * (1. - self.p)


@distribution.register_kl(OneHotCategorical, OneHotCategorical)
def _kl_one_hot_categorical_one_hot_categorical(dist1, dist2):
    return sum_mod.sum(dist1.p * (dist1.log_p - dist2.log_p), axis=-1)
