import chainer
from chainer import Distribution
from chainer.functions.noise.gaussian import gaussian
from chainer.functions.math.exponential import exp
import numpy


class Normal(Distribution):
    def __init__(self, mean, ln_var):
        """

        Args:
            mean(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Parameter of distribution representing the
            mean :math:`\\mu`.
            ln_var(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Parameter of distribution representing the
            logarithm of a variance :math:`\\log(\\sigma^2)`.
        """
        self.mean, self.ln_var = mean, ln_var

    def log_prob(self, x):
        return -0.5 * numpy.pi - 0.5 * self.ln_var \
               - 0.5 * (x - self.mean) ** 2 / exp(self.ln_var)

    def sample(self):
        return gaussian(self.mean, self.ln_var)
