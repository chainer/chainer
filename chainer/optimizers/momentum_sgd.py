from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class MomentumSGDHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of classical momentum SGD.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        momentum = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: MomentumSGDHyperparameter # NOQA
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
    _kernel = None

    def __init__(self, parent_hyperparam=None, lr=None, momentum=None):
        super(MomentumSGDRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if momentum is not None:
            self.hyperparam.momentum = momentum

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['v'] = xp.zeros_like(param.data)

        # For iDeep
        if isinstance(param.data, intel64.mdarray):
            self.state['v'] = intel64.ideep.array(
                self.state['v'], itype=intel64.ideep.wgt_array)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        v = self.state['v']
        if isinstance(v, intel64.mdarray):
            v.inplace_axpby(self.hyperparam.momentum, -
                            self.hyperparam.lr, grad)
            param.data += v
        else:
            v *= self.hyperparam.momentum
            v -= self.hyperparam.lr * grad
            param.data += v

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if MomentumSGDRule._kernel is None:
            MomentumSGDRule._kernel = cuda.elementwise(
                'T grad, T lr, T momentum',
                'T param, T v',
                '''v = momentum * v - lr * grad;
                   param += v;''',
                'momentum_sgd')
        MomentumSGDRule._kernel(
            grad, self.hyperparam.lr, self.hyperparam.momentum, param.data,
            self.state['v'])


class MomentumSGD(optimizer.GradientMethod):

    """Momentum SGD optimizer.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum):
        super(MomentumSGD, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')

    def create_update_rule(self):
        return MomentumSGDRule(self.hyperparam)
