import numpy

from chainer.backends import cuda
from chainer import optimizer


class AdaGradRule(optimizer.UpdateRule):

    """Update rule of AdaGrad.

    See :class:`~chainer.optimizers.AdaGrad` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.

    """

    def __init__(self, parent_hyperparam=None):
        super(AdaGradRule, self).__init__(parent_hyperparam)

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['h'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return

        lr = self.hyperparam.lr
        eps = self.hyperparam.eps
        h = self.state['h']

        h += grad * grad
        param.data -= lr * grad / (numpy.sqrt(h) + eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T h',
            '''h += grad * grad;
               param -= lr * grad / (sqrt(h) + eps);''',
            'adagrad')(grad, self.hyperparam.lr, self.hyperparam.eps,
                       param.data, self.state['h'])


class AdaGrad(optimizer.GradientMethod):

    """AdaGrad optimizer.

    See: http://jmlr.org/papers/v12/duchi11a.html

    Args:
        lr (float): Learning rate.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=0.001,
                 eps=1e-8, model=None):
        super(AdaGrad, self).__init__(model)
        self.hyperparam.lr = lr
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return AdaGradRule(self.hyperparam)
