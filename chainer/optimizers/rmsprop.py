import numpy

from chainer.backends import cuda
from chainer import optimizer


class RMSpropRule(optimizer.UpdateRule):

    """Update rule for RMSprop.

    See :class:`~chainer.optimizers.RMSprop` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.

    """

    def __init__(self, parent_hyperparam=None):
        super(RMSpropRule, self).__init__(parent_hyperparam)

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        ms = self.state['ms']

        ms *= hp.alpha
        ms += (1 - hp.alpha) * grad * grad
        param.data -= hp.lr * grad / (numpy.sqrt(ms) + eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / (sqrt(ms) + eps);''',
            'rmsprop')(grad, hp.lr, hp.alpha,
                       eps, param.data, self.state['ms'])


class RMSprop(optimizer.GradientMethod):

    """RMSprop optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=0.01,
                 alpha=0.99, eps=1e-8,
                 model=None):
        super(RMSprop, self).__init__(model)
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return RMSpropRule(self.hyperparam)
