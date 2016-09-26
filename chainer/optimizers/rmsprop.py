import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, alpha, eps, param, ms):
    ms *= alpha
    ms += (1 - alpha) * grad * grad
    param -= lr * grad / (cuda.sqrt_fixed(ms) + eps)


class RMSprop(optimizer.GradientMethod):

    """Hinton's RMSprop."""

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['ms'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        update(param.grad, numpy.float32(self.lr), numpy.float32(self.alpha),
               numpy.float32(self.eps), param.data, state['ms'])
