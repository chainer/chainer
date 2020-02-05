import numpy

import chainer
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class AdaGradHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of AdaGrad.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        eps = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: AdaGradHyperparameter # NOQA
_default_hyperparam.lr = 0.001
_default_hyperparam.eps = 1e-8


class AdaGradRule(optimizer.UpdateRule):

    """Update rule of AdaGrad.

    See :class:`~chainer.optimizers.AdaGrad` for the default values of the
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
        super(AdaGradRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        with chainer.using_device(param.device):
            self.state['h'] = param.device.xp.zeros_like(param.data)

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
        if AdaGradRule._kernel is None:
            AdaGradRule._kernel = cuda.elementwise(
                'T grad, T lr, T eps',
                'T param, T h',
                '''h += grad * grad;
                   param -= lr * grad / (sqrt(h) + eps);''',
                'adagrad')
        AdaGradRule._kernel(grad, self.hyperparam.lr, self.hyperparam.eps,
                            param.data, self.state['h'])


class AdaGrad(optimizer.GradientMethod):

    """AdaGrad optimizer.

    See: http://jmlr.org/papers/v12/duchi11a.html

    Args:
        lr (float): Learning rate.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=_default_hyperparam.lr, eps=_default_hyperparam.eps):
        super(AdaGrad, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return AdaGradRule(self.hyperparam)
