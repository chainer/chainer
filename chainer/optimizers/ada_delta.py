import numpy

import chainer
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class AdaDeltaHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of Zeiler's ADADELTA.

        This is only for PEP 544 compliant static type checkers.
        """
        rho = None  # type: float
        eps = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: AdaDeltaHyperparameter # NOQA
_default_hyperparam.rho = 0.95
_default_hyperparam.eps = 1e-6


class AdaDeltaRule(optimizer.UpdateRule):

    """Update rule of Zeiler's ADADELTA.

    See :class:`~chainer.optimizers.AdaDelta` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        rho (float): Exponential decay rate of the first and second order
            moments.
        eps (float): Small value for the numerical stability.

    """
    is_elementwise = True

    _kernel = None

    def __init__(self, parent_hyperparam=None, rho=None, eps=None):
        super(AdaDeltaRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if rho is not None:
            self.hyperparam.rho = rho
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        with chainer.using_device(param.device):
            xp = param.device.xp
            self.state['msg'] = xp.zeros_like(param.data)
            self.state['msdx'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        msg, msdx = self.state['msg'], self.state['msdx']
        rho = self.hyperparam.rho
        eps = self.hyperparam.eps

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = numpy.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if AdaDeltaRule._kernel is None:
            AdaDeltaRule._kernel = cuda.elementwise(
                'T grad, T one_minus_rho, T eps',
                'T param, T msg, T msdx',
                '''msg   = msg + one_minus_rho * (grad * grad - msg);
                   T dx  = sqrt((msdx + eps) / (msg + eps)) * grad;
                   msdx  += one_minus_rho * (dx * dx - msdx);
                   param -= dx;''',
                'adadelta')
        AdaDeltaRule._kernel(
            grad, 1 - self.hyperparam.rho, self.hyperparam.eps, param.data,
            self.state['msg'], self.state['msdx'])


class AdaDelta(optimizer.GradientMethod):

    """Zeiler's ADADELTA.

    See: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    Args:
        rho (float): Exponential decay rate of the first and second order
            moments.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, rho=_default_hyperparam.rho,
                 eps=_default_hyperparam.eps):
        super(AdaDelta, self).__init__()
        self.hyperparam.rho = rho
        self.hyperparam.eps = eps

    rho = optimizer.HyperparameterProxy('rho')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return AdaDeltaRule(self.hyperparam)
