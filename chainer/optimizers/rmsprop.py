import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class RMSpropHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of RMSprop.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        alpha = None  # type: float
        eps = None  # type: float
        eps_inside_sqrt = None  # type: bool


_default_hyperparam = optimizer.Hyperparameter()  # type: RMSpropHyperparameter # NOQA
_default_hyperparam.lr = 0.01
_default_hyperparam.alpha = 0.99
_default_hyperparam.eps = 1e-8
_default_hyperparam.eps_inside_sqrt = False


class RMSpropRule(optimizer.UpdateRule):

    """Update rule for RMSprop.

    See :class:`~chainer.optimizers.RMSprop` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eps_inside_sqrt (bool): When ``True``, gradient will be divided by
            :math:`\\sqrt{ms + eps}` where ``ms`` is the mean square. When
            ``False`` (default), gradient will be divided by
            :math:`\\sqrt{ms} + eps` instead.
            This option may be convenient for users porting code from other
            frameworks;
            see `#4754 <https://github.com/chainer/chainer/issues/4754>`__ for
            details.

    """

    def __init__(self, parent_hyperparam=None, lr=None, alpha=None, eps=None,
                 eps_inside_sqrt=None):
        super(RMSpropRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if eps is not None:
            self.hyperparam.eps = eps
        if eps_inside_sqrt is not None:
            self.hyperparam.eps_inside_sqrt = eps_inside_sqrt

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        ms = self.state['ms']

        ms *= hp.alpha
        ms += (1 - hp.alpha) * grad * grad
        if hp.eps_inside_sqrt:
            denom = numpy.sqrt(ms + eps)
        else:
            denom = numpy.sqrt(ms) + eps
        param.data -= hp.lr * grad / denom

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        if hp.eps_inside_sqrt:
            denom = 'sqrt(ms + eps)'
        else:
            denom = 'sqrt(ms) + eps'
        kernel = cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / ({});'''.format(denom),
            'rmsprop')
        kernel(grad, self.hyperparam.lr, self.hyperparam.alpha,
               eps, param.data, self.state['ms'])


class RMSprop(optimizer.GradientMethod):

    """RMSprop optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eps_inside_sqrt (bool): When ``True``, gradient will be divided by
            :math:`\\sqrt{ms + eps}` where ``ms`` is the mean square. When
            ``False`` (default), gradient will be divided by
            :math:`\\sqrt{ms} + eps` instead.
            This option may be convenient for users porting code from other
            frameworks;
            see `#4754 <https://github.com/chainer/chainer/issues/4754>`__ for
            details.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 alpha=_default_hyperparam.alpha, eps=_default_hyperparam.eps,
                 eps_inside_sqrt=_default_hyperparam.eps_inside_sqrt):
        super(RMSprop, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.eps = eps
        self.hyperparam.eps_inside_sqrt = eps_inside_sqrt

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    eps = optimizer.HyperparameterProxy('eps')
    eps_inside_sqrt = optimizer.HyperparameterProxy('eps_inside_sqrt')

    def create_update_rule(self):
        return RMSpropRule(self.hyperparam)
