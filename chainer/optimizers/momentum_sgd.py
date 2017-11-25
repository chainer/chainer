from chainer import cuda
from chainer.numexpr_config import numexpr
from chainer.numexpr_config import numexpr_enabled
from chainer import optimizer

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9


class MomentumSGDRule(optimizer.UpdateRule):

    """Update rule for the classical momentum SGD.

    See :class:`~chainer.optimizers.MomentumSGD` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, parent_hyperparam=None, lr=None, momentum=None):
        super(MomentumSGDRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if momentum is not None:
            self.hyperparam.momentum = momentum

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        v = self.state['v']
        if numexpr_enabled:
            momentum, lr = self.hyperparam.momentum, self.hyperparam.lr  # NOQA
            data = param.data
            numexpr.evaluate('v*momentum - lr*grad',
                             out=v, casting='same_kind')
            numexpr.evaluate('data + v', out=data, casting='same_kind')
        else:
            v *= self.hyperparam.momentum
            v -= self.hyperparam.lr * grad
            param.data += v

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(
                grad, self.hyperparam.lr, self.hyperparam.momentum,
                param.data, self.state['v'])


class MomentumSGD(optimizer.GradientMethod):

    """Momentum SGD optimizer.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum, model=None):
        super(MomentumSGD, self).__init__(model)
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')

    def create_update_rule(self):
        return MomentumSGDRule(self.hyperparam)
