import chainer
from chainer.functions.math import exponential


class Bijector(object):

    """Interface of Bijector.

    `Bijector` is implementation of bijective (invertible) function that is
    used by `TransformedDistribution`. The three method `_forward`, `_inv` and
    `_logdet_jac` have to be defined in inhereted class.
    """

    def __init__(self):
        self.cache_x_y = (None, None)

    def forward_(self, x):
        old_x, old_y = self.cache_x_y
        if x is old_x:
            return old_y
        y = self.forward(x)
        self.cache_x_y = x, y
        return y

    def inv_(self, y):
        old_x, old_y = self.cache_x_y
        if y is old_y:
            return old_x
        x = self.inv(y)
        self.cache_x_y = x, y
        return x

    def logdet_jac_(self, x):
        return self.logdet_jac(x)

    def forward(self, x):
        """Forward computation

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            based distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            transformed distribution.
        """
        raise NotImplementedError

    def inv(self, y):
        """Inverse computation

        Args:
            y(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            transformed distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            based distribution.
        """
        raise NotImplementedError

    def logdet_jac(self, x):
        """Log-Determinant of Jacobian matrix of transformation

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            based distribution.

        Returns:
            ~chainer.Variable: Log-Determinant of Jacobian matrix in `x`.
        """
        raise NotImplementedError


class ExpBijector(Bijector):

    """ExpBijector.

    """

    def forward(self, x):
        return exponential.exp(x)

    def inv(self, y):
        return exponential.log(y)

    def logdet_jac(self, x):
        return x
