import numpy

from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 1e-4
_default_hyperparam.alpha = 0.95
_default_hyperparam.momentum = 0.9
_default_hyperparam.eps = 1e-4


class RMSpropGravesRule(optimizer.UpdateRule):

    """Update rule for Alex Graves's RMSprop.

    See :class:`~chainer.optimizers.RMSpropGraves` for the default values of
    the hyperparameters.

    Args:
        parent_hyperparam (~chainer.Hyperparameter): Hyperparameter that
            provides the default values.
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the first and second order
            moments of the raw gradient.
        momentum (float): Exponential decay rate of the first order moment of
            the adjusted gradient.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, parent_hyperparam=None,
                 lr=None, alpha=None, momentum=None, eps=None):
        super(RMSpropGravesRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if momentum is not None:
            self.hyperparam.momentum = momentum
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['n'] = xp.zeros_like(param.data)
            self.state['g'] = xp.zeros_like(param.data)
            self.state['delta'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        n, g, delta = self.state['n'], self.state['g'], self.state['delta']
        hp = self.hyperparam

        n *= hp.alpha
        n += (1 - hp.alpha) * grad * grad
        g *= hp.alpha
        g += (1 - hp.alpha) * grad
        delta *= hp.momentum
        delta -= hp.lr * grad / numpy.sqrt(n - g * g + hp.eps)
        param.data += delta

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        cuda.elementwise(
            'T grad, T lr, T alpha, T momentum, T eps',
            'T param, T avg_n, T avg_g, T delta',
            '''avg_n = alpha * avg_n + (1 - alpha) * grad * grad;
               avg_g = alpha * avg_g + (1 - alpha) * grad;
               delta = delta * momentum -
                   lr * grad * rsqrt(avg_n - avg_g * avg_g + eps);
               param += delta;''',
            'rmsprop_graves')(
                grad, hp.lr, hp.alpha, hp.momentum, hp.eps, param.data,
                self.state['n'], self.state['g'], self.state['delta'])


class RMSpropGraves(optimizer.GradientMethod):

    """Alex Graves's RMSprop.

    See: http://arxiv.org/abs/1308.0850

    Args:
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the first and second order
            moments of the raw gradient.
        momentum (float): Exponential decay rate of the first order moment of
            the adjusted gradient.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 alpha=_default_hyperparam.alpha,
                 momentum=_default_hyperparam.momentum,
                 eps=_default_hyperparam.eps):
        super(RMSpropGraves, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.momentum = momentum
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    momentum = optimizer.HyperparameterProxy('momentum')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return RMSpropGravesRule(self.hyperparam)
