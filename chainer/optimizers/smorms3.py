import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, eps, param, mem, g, g2):
    r = 1 / (mem + 1)
    g *= 1 - r
    g += r * grad
    g2 *= 1 - r
    g2 += r * grad * grad
    x = g * g / (g2 + eps)
    param -= grad * cuda.minimum(lr, x) / (cuda.sqrt_fixed(g2) + eps)
    mem *= 1 - x
    mem += 1


class SMORMS3(optimizer.GradientMethod):

    """Simon Funk's SMORMS3.

    See http://sifter.org/~simon/journal/20150420.html.

    """

    def __init__(self, lr=0.001, eps=1e-16):
        self.lr = lr
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['mem'] = xp.ones_like(param.data)
            state['g'] = xp.zeros_like(param.data)
            state['g2'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        update(param.grad, numpy.float32(self.lr), numpy.float32(self.eps),
               param.data, state['mem'], state['g'], state['g2'])
