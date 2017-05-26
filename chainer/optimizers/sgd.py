from chainer import cuda
from chainer import optimizer


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param, state):
        param.data -= self.lr * param.grad
