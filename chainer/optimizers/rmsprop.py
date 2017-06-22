import numpy

from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.alpha = 0.99
_default_hyperparam.eps = 1e-8


class RMSpropRule(optimizer.UpdateRule):

    """Update rule for RMSprop.

    See :class:`~chainer.optimizers.RMSprop` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.Hyperparameter): Hyperparameter that
            provides the default values.
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, parent_hyperparam=None, lr=None, alpha=None, eps=None):
        super(RMSpropRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        ms = self.state['ms']

        ms *= hp.alpha
        ms += (1 - hp.alpha) * grad * grad
        param.data -= hp.lr * grad / (numpy.sqrt(ms) + hp.eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / (sqrt(ms) + eps);''',
            'rmsprop')(grad, self.hyperparam.lr, self.hyperparam.alpha,
                       self.hyperparam.eps, param.data, self.state['ms'])


class RMSprop(optimizer.GradientMethod):

    """RMSprop optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 alpha=_default_hyperparam.alpha, eps=_default_hyperparam.eps):
        super(RMSprop, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return RMSpropRule(self.hyperparam)
