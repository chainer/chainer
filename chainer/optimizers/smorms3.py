import numpy

import chainer
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class SMORMS3Hyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of Simon Funk's SMORMS3.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        eps = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: SMORMS3Hyperparameter # NOQA
_default_hyperparam.lr = 0.001
_default_hyperparam.eps = 1e-16


class SMORMS3Rule(optimizer.UpdateRule):

    """Update rule for Simon Funk's SMORMS3.

    See :class:`~chainer.optimizers.SMORMS3` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        eps (float): Small value for the numerical stability.

    """
    is_elementwise = True

    _kernel = None

    def __init__(self, parent_hyperparam=None, lr=None, eps=None):
        super(SMORMS3Rule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        with chainer.using_device(param.device):
            xp = param.device.xp
            self.state['mem'] = xp.ones_like(param.data)
            self.state['g'] = xp.zeros_like(param.data)
            self.state['g2'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        mem, g, g2 = self.state['mem'], self.state['g'], self.state['g2']

        r = 1 / (mem + 1)
        g = (1 - r) * g + r * grad
        g2 = (1 - r) * g2 + r * grad * grad
        x = g * g / (g2 + self.hyperparam.eps)
        param.data -= grad * numpy.minimum(x, self.hyperparam.lr) \
            / (numpy.sqrt(g2) + self.hyperparam.eps)
        mem = 1 + mem * (1 - x)

        self.state['mem'], self.state['g'], self.state['g2'] = mem, g, g2

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if SMORMS3Rule._kernel is None:
            SMORMS3Rule._kernel = cuda.elementwise(
                'T grad, T lr, T eps',
                'T param, T mem, T g, T g2',
                '''T r, x;
                   r = 1 / (mem + 1);
                   g = (1 - r) * g + r * grad;
                   g2 = (1 - r) * g2 + r * grad * grad;
                   x = g * g / (g2 + eps);
                   param -= grad * min(lr, x) / (sqrt(g2) + eps);
                   mem = 1 + mem * (1 - x)
                   ''',
                'smorms3')
        SMORMS3Rule._kernel(
            grad, self.hyperparam.lr, self.hyperparam.eps, param.data,
            self.state['mem'], self.state['g'], self.state['g2'])


class SMORMS3(optimizer.GradientMethod):

    """Simon Funk's SMORMS3.

    See http://sifter.org/~simon/journal/20150420.html.

    Args:
        lr (float): Learning rate.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=_default_hyperparam.lr, eps=_default_hyperparam.eps):
        super(SMORMS3, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return SMORMS3Rule(self.hyperparam)
