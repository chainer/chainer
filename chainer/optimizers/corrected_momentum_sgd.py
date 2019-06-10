from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class CorrectedMomentumSGDHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of corrected momentum SGD.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        momentum = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: CorrectedMomentumSGDHyperparameter # NOQA
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9


class CorrectedMomentumSGDRule(optimizer.UpdateRule):

    """Update rule for the corrected momentum SGD.

    See :class:`~chainer.optimizers.CorrectedMomentumSGD` for the default
    values of the hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, parent_hyperparam=None, lr=None, momentum=None):
        super(CorrectedMomentumSGDRule, self).__init__(
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
            v.inplace_axpby(self.hyperparam.momentum,
                            -1, grad)
            param.data += self.hyperparam.lr * v
        else:
            v *= self.hyperparam.momentum
            v -= grad
            param.data += self.hyperparam.lr * v

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - grad;
               param += lr * v;''',
            'momentum_sgd')(
                grad, self.hyperparam.lr, self.hyperparam.momentum,
                param.data, self.state['v'])


class CorrectedMomentumSGD(optimizer.GradientMethod):

    """Momentum SGD optimizer.

    This implements momentum correction discussed in the third section of
    `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour \
    <https://arxiv.org/abs/1706.02677>`_.

    :class:`~chainer.optimizers.MomentumSGD` implements the equation (10) of
    the paper. This optimizer implements the equation (9).

    To get better understanding between the two methods,
    we show the equivalence between the equation (9) and modification of
    the equation (10) that takes momentum correction into account.
    First, we set :math:`v_{t} = \\eta_{t} u_t`.
    We substitute this relation to the equation (10).

    .. math::

        v_{t+1} &= m\\frac{\\eta_{t+1}}{\\eta_{t}}v_t + \\eta_{t+1}g_t  \\\\
                &= m\\frac{\\eta_{t+1}}{\\eta_{t}}\\eta_{t}u_t +
                \\eta_{t+1}g_t \\\\
                &= \\eta_{t+1}(m u_t + g_t) \\\\

    From this result, we derive :math:`u_{t+1} = m u_t + g_t`, which is how
    update tensors are calculated by
    :class:`~chainer.optimizers.CorrectedMomentumSGD`. Thus, the equivalence
    is shown.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum):
        super(CorrectedMomentumSGD, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')

    def create_update_rule(self):
        return CorrectedMomentumSGDRule(self.hyperparam)
