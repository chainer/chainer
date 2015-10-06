import math

import numpy

from chainer import cuda
from chainer import optimizer


class Adam(optimizer.GradientMethod):

    """Adam optimization algorithm.

    See: http://arxiv.org/abs/1412.6980v8

    """
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        optimizer.GradientMethod.__init__(self)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param)
        state['m'] = xp.zeros_like(param)
        state['v'] = xp.zeros_like(param)

    def update_param_cpu(self, param, grad, state):
        m, v = state['m'], state['v']
        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param -= self.lr * m / (numpy.sqrt(v) + self.eps)

    def update_param_gpu(self, param, grad, state):
        cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (sqrt(v) + eps);''',
            'adam')(grad, self.lr, 1 - self.beta1, 1 - self.beta2, self.eps,
                    param, state['m'], state['v'])

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * math.sqrt(fix2) / fix1
