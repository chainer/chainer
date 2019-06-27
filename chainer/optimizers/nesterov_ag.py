import chainer
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class NesterovAGHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of Nesterov's Accelerated Gradient.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        momentum = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: NesterovAGHyperparameter # NOQA
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9


class NesterovAGRule(optimizer.UpdateRule):

    """Update rule for Nesterov's Accelerated Gradient.

    See :class:`~chainer.optimizers.NesterovAG` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """
    is_elementwise = True

    _kernel = None

    def __init__(self, parent_hyperparam=None, lr=None, momentum=None):
        super(NesterovAGRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if momentum is not None:
            self.hyperparam.momentum = momentum

    def init_state(self, param):
        with chainer.using_device(param.device):
            xp = param.device.xp
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        v = self.state['v']
        lr, momentum = self.hyperparam.lr, self.hyperparam.momentum

        v *= momentum
        v -= lr * grad
        param.data += momentum * momentum * v
        param.data -= (1 + momentum) * lr * grad

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if NesterovAGRule._kernel is None:
            NesterovAGRule._kernel = cuda.elementwise(
                'T grad, T lr, T momentum',
                'T param, T v',
                '''
                v = v * momentum - lr * grad;
                param += momentum * momentum * v - (1 + momentum) * lr * grad;
                ''',
                'nesterov_ag')
        NesterovAGRule._kernel(
            grad, self.hyperparam.lr, self.hyperparam.momentum,
            param.data, self.state['v'])


class NesterovAG(optimizer.GradientMethod):

    """Nesterov's Accelerated Gradient.

    See: https://arxiv.org/abs/1212.0901

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum):
        super(NesterovAG, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')

    def create_update_rule(self):
        return NesterovAGRule(self.hyperparam)
