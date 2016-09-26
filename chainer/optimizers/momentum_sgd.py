import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, momentum, param, v):
    v *= momentum
    v -= lr * grad
    param += v


class MomentumSGD(optimizer.GradientMethod):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        update(param.grad,
               numpy.float32(self.lr), numpy.float32(self.momentum),
               param.data, state['v'])
