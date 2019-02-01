from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class SGDHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of vanilla stochastic gradient descent.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: SGDHyperparameter # NOQA
_default_hyperparam.lr = 0.01


class SGDRule(optimizer.UpdateRule):

    """Update rule of vanilla stochastic gradient descent.

    See :class:`~chainer.optimizers.SGD` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.

    """
    _kernel = None

    def __init__(self, parent_hyperparam=None, lr=None):
        super(SGDRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if isinstance(param.data, intel64.mdarray):
            param.data.inplace_axpby(1.0, -self.hyperparam.lr, grad)
        else:
            param.data -= self.hyperparam.lr * grad

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if SGDRule._kernel is None:
            SGDRule._kernel = cuda.elementwise(
                'T grad, T lr', 'T param',
                'param -= lr * grad', 'sgd')
        SGDRule._kernel(grad, self.hyperparam.lr, param.data)


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
