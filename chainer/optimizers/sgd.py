import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, param):
    param -= lr * grad


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param, state):
        update(param.grad, numpy.float32(self.lr), param.data)
