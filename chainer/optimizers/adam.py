import math

import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, one_minus_beta1, one_minus_beta2, eps, param, m, v):
    m += one_minus_beta1 * (grad - m)
    v += one_minus_beta2 * (grad * grad - v)
    param -= lr * m / (cuda.sqrt_fixed(v) + eps)


class Adam(optimizer.GradientMethod):

    """Adam optimization algorithm.

    See: http://arxiv.org/abs/1412.6980v8

    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['m'] = xp.zeros_like(param.data)
            state['v'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        update(param.grad, numpy.float32(self.lr),
               numpy.float32(1 - self.beta1), numpy.float32(1 - self.beta2),
               numpy.float32(self.eps), param.data, state['m'], state['v'])

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * math.sqrt(fix2) / fix1
