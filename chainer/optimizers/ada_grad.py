import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, eps, param, h):
    h += grad * grad
    param -= lr * grad / (cuda.sqrt_fixed(h) + eps)


class AdaGrad(optimizer.GradientMethod):

    """AdaGrad implementation.

    See: http://jmlr.org/papers/v12/duchi11a.html

    """

    def __init__(self, lr=0.001, eps=1e-8):
        self.lr = lr
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['h'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        update(param.grad, numpy.float32(self.lr), numpy.float32(self.eps),
               param.data, state['h'])
