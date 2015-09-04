import numpy

from chainer import cuda
from chainer import optimizer


class RMSprop(optimizer.GradientMethod):

    """Hinton's RMSprop."""

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        optimizer.Optimizer.__init__(self)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param)
        state['ms'] = xp.zeros_like(param)

    def update_param_cpu(self, param, grad, state):
        ms = state['ms']
        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param -= self.lr * grad / (numpy.sqrt(ms) + self.eps)

    def update_param_gpu(self, param, grad, state):
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / (sqrt(ms) + eps);''',
            'rmsprop')(grad, self.lr, self.alpha, self.eps,
                       param, state['ms'])
