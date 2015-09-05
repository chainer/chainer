import numpy

from chainer import cuda
from chainer import optimizer


class AdaDelta(optimizer.GradientMethod):

    """Zeiler's ADADELTA.

    See: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    """
    def __init__(self, rho=0.95, eps=1e-6):
        optimizer.GradientMethod.__init__(self)
        self.rho = rho
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param)
        state['msg'] = xp.zeros_like(param)
        state['msdx'] = xp.zeros_like(param)

    def update_param_cpu(self, param, grad, state):
        msg, msdx = state['msg'], state['msdx']
        msg *= self.rho
        msg += (1 - self.rho) * grad * grad
        dx = numpy.sqrt((msdx + self.eps) / (msg + self.eps)) * grad
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx
        param -= dx

    def update_param_gpu(self, param, grad, state):
        cuda.elementwise(
            'T grad, T one_minus_rho, T eps',
            'T param, T msg, T msdx',
            '''msg   = msg + one_minus_rho * (grad * grad - msg);
               T dx  = sqrt((msdx + eps) / (msg + eps)) * grad;
               msdx  += one_minus_rho * (dx * dx - msdx);
               param -= dx;''',
            'adadelta')(grad, 1 - self.rho, self.eps,
                        param, state['msg'], state['msdx'])
