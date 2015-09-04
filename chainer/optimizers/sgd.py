from chainer import cuda
from chainer import optimizer


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        optimizer.Optimizer.__init__(self)
        self.lr = lr

    def update_param_cpu(self, param, grad, _):
        param -= self.lr * grad

    def update_param_gpu(self, param, grad, _):
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(grad, self.lr, param)
