from chainer.functions.math import exponential
from chainer.functions.math import identity


class Bijector():
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
        raise NotImplementedError

    def _inv(self, y):
        raise NotImplementedError

    def _logdet_jac(self, x):
        raise NotImplementedError


class ExpBijector(Bijector):
    def _forward(self, x):
        return exponential.exp(x)

    def _inv(self, y):
        return exponential.log(y)

    def _logdet_jac(self, x):
        return identity.identity(x)
