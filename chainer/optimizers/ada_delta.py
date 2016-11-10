import numpy

from chainer import cuda
from chainer import optimizer


@cuda.fuse()
def update(grad, one_minus_rho, eps, param, msg, msdx):
    msg += one_minus_rho * (grad * grad - msg)
    dx = cuda.sqrt_fixed((msdx + eps) / (msg + eps)) * grad
    msdx += one_minus_rho * (dx * dx - msdx)
    param -= dx


class AdaDelta(optimizer.GradientMethod):

    """Zeiler's ADADELTA.

    See: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    """

    def __init__(self, rho=0.95, eps=1e-6):
        self.rho = rho
        self.eps = eps

    def init_state(self, param, state):
        data = param.data
        xp = cuda.get_array_module(data)
        with cuda.get_device(data):
            state['msg'] = xp.zeros_like(data)
            state['msdx'] = xp.zeros_like(data)

    def update_one(self, param, state):
        update(param.grad,
               numpy.float32(1 - self.rho), numpy.float32(self.eps),
               param.data, state['msg'], state['msdx'])
