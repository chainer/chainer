import chainer
from chainer.backends import cuda
from chainer import Distribution
from chainer.functions.array import broadcast
from chainer.functions.array import expand_dims
from chainer.functions.array import repeat
from chainer.functions.array import rollaxis
from chainer.functions.array import squeeze
from chainer.functions.array import swapaxes
from chainer.functions.math import basic_math
from chainer.functions.math import exponential
from chainer.functions.math import inv
from chainer.functions.math import matmul
from chainer.functions.math import sum
import numpy


class MultivariateNormal(Distribution):

    """MultivariateNormal Distribution.

    Args:
        loc(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        location :math:`\\mu`.
        scale_tril(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Parameter of distribution representing the \
        scale :math:`L`.

    """

    def __init__(self, loc, scale_tril):
        if isinstance(loc, chainer.Variable):
            self.loc = loc
        else:
            self.loc = chainer.Variable(loc)
        if isinstance(scale_tril, chainer.Variable):
            self.scale_tril = scale_tril
        else:
            self.scale_tril = chainer.Variable(scale_tril)
        self.d = self.scale_tril.shape[-1]

    def __copy__(self):
        return self._copy_to(MultivariateNormal(self.loc, self.scale_tril))

    @property
    def batch_shape(self):
        return self.loc.shape[:-1]

    @property
    def entropy(self):
        """Returns entropy.

        Returns:
            Output Variable representing entropy.

        """
        st = rollaxis.rollaxis(self.scale_tril, -2, 0)
        st = rollaxis.rollaxis(st, -1, 1)
        diag = st[list(range(self.d)), list(range(self.d))]
        return sum.sum(exponential.log(basic_math.absolute(diag)), axis=0) \
            + 0.5 * numpy.log(2 * numpy.pi * numpy.e) * self.d

    @property
    def event_shape(self):
        return self.loc.shape[-1:]

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
        st = rollaxis.rollaxis(self.scale_tril, -2, 0)
        st = rollaxis.rollaxis(st, -1, 1)
        diag = st[list(range(self.d)), list(range(self.d))]
        logdet = sum.sum(exponential.log(basic_math.absolute(diag)), axis=0)
        scale_tril_inv = \
            inv.batch_inv(self.scale_tril.reshape(-1, self.d, self.d)).reshape(
                self.scale_tril.shape)
        scale_tril_inv = scale_tril_inv.reshape(
            self.batch_shape+(self.d, self.d))
        print(scale_tril_inv.shape)

        bsti = broadcast.broadcast_to(scale_tril_inv, x.shape + (self.d,))
        bl = broadcast.broadcast_to(self.loc, x.shape)
        m = matmul.matmul(
            bsti,
            expand_dims.expand_dims(x - bl, axis=-1))
        m = matmul.matmul(swapaxes.swapaxes(m, -1, -2), m)
        m = squeeze.squeeze(m, axis=-1)
        m = squeeze.squeeze(m, axis=-1)
        logz = - 0.5 * self.d * numpy.log(2 * numpy.pi) - logdet
        return broadcast.broadcast_to(logz, m.shape) - 0.5 * m

    @property
    def mean(self):
        """Returns mean value.

        Returns:
            ~chainer.Variable: Output variable representing mean value.

        """
        return self.loc

    @property
    def mode(self):
        """Returns mode.

        Returns:
            ~chainer.Variable: Output variable representing mode.

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
            eps = cuda.cupy.random.standard_normal(
                (n,)+self.loc.shape+(1,), dtype=self.loc.dtype)
        else:
            eps = numpy.random.standard_normal(
                (n,)+self.loc.shape+(1,)).astype(numpy.float32)

        noise = matmul.matmul(repeat.repeat(
            expand_dims.expand_dims(self.scale_tril, axis=0), n, axis=0), eps)
        noise = squeeze.squeeze(noise, axis=-1)
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
