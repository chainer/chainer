import copy

from chainer import backend
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


class AdaDeltaKernel(object):
    def __init__(self, hp):
        self.hp = copy.copy(hp)

        @cuda.fuse(kernel_name='adadelta')
        def kernel(rho, eps, grad, param, msg, msdx):
            xp = cuda.get_array_module(grad)
            msg *= rho
            msg += (1 - rho) * grad * grad
            dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
            msdx *= rho
            msdx += (1 - rho) * dx * dx
            param -= dx

        @cuda.fuse(kernel_name='adadelta')
        def kernel_const_hp(*args):
            kernel(hp.rho, hp.eps, *args)

        self.kernel_nonconst_hp = kernel
        self.kernel_const_hp = kernel_const_hp

    def __call__(self, hp, grad, param, msg, msdx):
        if hp.get_dict() == self.hp:
            self.kernel_const_hp(grad, param, msg, msdx)
        else:
            self.kernel_nonconst_hp(hp.rho, hp.eps, grad, param, msg, msdx)


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
    _kernel = None

    def __init__(self, parent_hyperparam=None, rho=None, eps=None):
        super(AdaDeltaRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if rho is not None:
            self.hyperparam.rho = rho
        if eps is not None:
            self.hyperparam.eps = eps
        self.kernel = AdaDeltaKernel(self.hyperparam)

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['msg'] = xp.zeros_like(param.data)
            self.state['msdx'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        self.kernel(self.hyperparam, grad, param.data,
                    self.state['msg'], self.state['msdx'])

    def update_core_gpu(self, param):
        self.update_core_cpu(param)


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
