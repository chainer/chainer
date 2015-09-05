import numpy

from chainer import cuda
from chainer import optimizer


class MomentumSGD(optimizer.GradientMethod):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        optimizer.GradientMethod.__init__(self)
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param)
        state['v'] = xp.zeros_like(param)

    def update_param_cpu(self, param, grad, state):
        v = state['v']
        v *= self.momentum
        v -= self.lr * grad
        param += v

    def update_param_gpu(self, param, grad, state):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(grad, self.lr, self.momentum,
                            param, state['v'])
