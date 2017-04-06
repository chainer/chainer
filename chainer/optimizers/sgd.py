from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01


class SGDRule(optimizer.UpdateRule):

    """Update rule of vanilla stochastic gradient descent.

    See :class:`~chainer.optimizers.SGD` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.Hyperparameter): Hyperparameter that
            provides the default values.
        lr (float): Learning rate.

    """

    def __init__(self, parent_hyperparam=None, lr=None):
        super(SGDRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        param.data -= self.hyperparam.lr * grad

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(grad, self.hyperparam.lr, param.data)


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent.

    Args:
        lr (float): Learning rate.

    """

    def __init__(self, lr=_default_hyperparam.lr):
        super(SGD, self).__init__()
        self.hyperparam.lr = lr

    lr = optimizer.HyperparameterProxy('lr')

    def create_update_rule(self):
        return SGDRule(self.hyperparam)
