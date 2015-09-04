import numpy

from chainer import cuda
from chainer import optimizer


class AdaGrad(optimizer.GradientMethod):

    """AdaGrad implementation.

    See: http://jmlr.org/papers/v12/duchi11a.html

    """
    def __init__(self, lr=0.001, eps=1e-8):
        optimizer.Optimizer.__init__()
        self.lr = lr
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param)
        state['h'] = xp.zeros_like(param)

    def update_param_cpu(self, param, grad, state):
        h = state['h']
        h += grad * grad
        param -= self.lr * grad / (numpy.sqrt(h) + self.eps)

    def update_param_gpu(self, param, grad, state):
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T h',
            '''h += grad * grad;
               param -= lr * grad / (sqrt(h) + eps);''',
            'adagrad')(grad, self.lr, self.eps,
                       param, state['h'])
