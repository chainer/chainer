from chainer.functions.math import exponential
from chainer.functions.math import identity


class Bijector():

    """Interface of Bijector.

    `Bijector` is implementation of bijective (invertible) function that is
    used by `TransformedDistribution`. The three method `_forward`, `_inv` and
    `_logdet_jac` have to be defined in inhereted class.
    """

    def __init__(self):
        self.cache_x_y = (None, None)

    def forward(self, x):
        old_x, old_y = self.cache_x_y
        if id(x) == id(old_x):
            return old_y
        y = self._forward(x)
        self.cache_x_y = x, y
        return y

    def inv(self, y):
        old_x, old_y = self.cache_x_y
        if id(y) == id(old_y):
            return old_x
        x = self._inv(y)
        self.cache_x_y = x, y
        return x

    def logdet_jac(self, x):
        return self._logdet_jac(x)

    def _forward(self, x):
        """Forward computation

        Args:
            x(:class:`~chainer.Variable`): Data points in the domain of the
            based distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            transformed distribution.
        """
        raise NotImplementedError

    def _inv(self, y):
        """Inverse computation

        Args:
            x(:class:`~chainer.Variable`): Data points in the domain of the
            transformed distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            based distribution.
        """
        raise NotImplementedError

    def _logdet_jac(self, x):
        """Log-Determinant of Jacobian matrix of transformation

        Args:
            x(:class:`~chainer.Variable`): Data points in the domain of the
            based distribution.

        Returns:
            ~chainer.Variable: Log-Determinant of Jacobian matrix in `x`.
        """
        raise NotImplementedError


class ExpBijector(Bijector):

    """ExpBijector.

    """

    def _forward(self, x):
        return exponential.exp(x)

    def _inv(self, y):
        return exponential.log(y)

    def _logdet_jac(self, x):
        return identity.identity(x)
