import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, lr, momentum, param, v):
    v *= momentum
    v -= lr * grad
    param += momentum * momentum * v - (1 + momentum) * lr * grad


class NesterovAG(optimizer.GradientMethod):

    """Nesterov's Accelerated Gradient.

    Formulated as the linear combination coefficients of the velocity and
    gradient contributions at each iteration.

    See: http://arxiv.org/abs/1212.0901

    """

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
