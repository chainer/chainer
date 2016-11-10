import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, alpha, momentum, eps, param, avg_n, avg_g, delta):
    avg_n *= alpha
    avg_n += (1 - alpha) * grad * grad
    avg_g *= alpha
    avg_g += (1 - alpha) * grad
    delta *= momentum
    delta -= lr * grad / cuda.sqrt_fixed(avg_n - avg_g * avg_g + eps)
    param += delta


class RMSpropGraves(optimizer.GradientMethod):

    """Alex Graves's RMSprop.

    See http://arxiv.org/abs/1308.0850

    """

    def __init__(self, lr=1e-4, alpha=0.95, momentum=0.9, eps=1e-4):
        # Default parameter values are the ones in the original paper.
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['n'] = xp.zeros_like(param.data)
            state['g'] = xp.zeros_like(param.data)
            state['delta'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        update(param.grad, numpy.float32(self.lr), numpy.float32(self.alpha),
               numpy.float32(self.momentum), numpy.float32(self.eps),
               param.data, state['n'], state['g'], state['delta'])
