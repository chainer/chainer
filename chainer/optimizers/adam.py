from __future__ import division
import math
import warnings

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class AdamHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of Adam.

        This is only for PEP 544 compliant static type checkers.
        """
        alpha = None  # type: float
        beta1 = None  # type: float
        beta2 = None  # type: float
        eps = None  # type: float
        eta = None  # type: float
        weight_decay_rate = None  # type: float
        amsgrad = None  # type: bool


_default_hyperparam = optimizer.Hyperparameter()  # type: AdamHyperparameter # NOQA
_default_hyperparam.alpha = 0.001
_default_hyperparam.beta1 = 0.9
_default_hyperparam.beta2 = 0.999
_default_hyperparam.eps = 1e-8
_default_hyperparam.eta = 1.0
_default_hyperparam.weight_decay_rate = 0
_default_hyperparam.amsgrad = False


def _learning_rate(hp, t):
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Adam optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    return hp.alpha * math.sqrt(fix2) / fix1


class AdamRule(optimizer.UpdateRule):

    """Update rule of Adam optimization algorithm.

    See: `Adam: A Method for Stochastic Optimization \
          <https://arxiv.org/abs/1412.6980v8>`_

    Modified for proper weight decay.

    See: `Fixing Weight Decay Regularization in Adam \
          <https://openreview.net/forum?id=rk6qdGgCZ>`_

    With option to use AMSGrad variant of Adam.

    See: `On the Convergence of Adam and Beyond \
          <https://openreview.net/forum?id=ryQu7f-RZ>`_

    See :class:`~chainer.optimizers.Adam` for the default values
    of the hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.
        amsgrad (bool): Whether to use the AMSGrad variant of Adam.

    """
    _kernel = None
    _amsgrad_kernel = None

    def __init__(self, parent_hyperparam=None,
                 alpha=None, beta1=None, beta2=None, eps=None,
                 eta=None, weight_decay_rate=None, amsgrad=None):
        super(AdamRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if beta1 is not None:
            self.hyperparam.beta1 = beta1
        if beta2 is not None:
            self.hyperparam.beta2 = beta2
        if eps is not None:
            self.hyperparam.eps = eps
        if eta is not None:
            self.hyperparam.eta = eta
        if weight_decay_rate is not None:
            self.hyperparam.weight_decay_rate = weight_decay_rate
        if amsgrad is not None:
            self.hyperparam.amsgrad = amsgrad

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)
            if self.hyperparam.amsgrad:
                self.state['vhat'] = xp.zeros_like(param.data)

        # For iDeep
        if isinstance(param.data, intel64.mdarray):
            self.state['m'] = intel64.ideep.array(
                self.state['m'], itype=intel64.ideep.wgt_array)
            self.state['v'] = intel64.ideep.array(
                self.state['v'], itype=intel64.ideep.wgt_array)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        m, v = self.state['m'], self.state['v']
        if (isinstance(m, intel64.mdarray)
                and isinstance(v, intel64.mdarray)):
            m.inplace_axpby(1.0, 1.0 - hp.beta1, grad - m)
            v.inplace_axpby(1.0, 1.0 - hp.beta2, grad*grad - v)
            if hp.amsgrad:
                vhat = self.state['vhat']
                numpy.maximum(vhat, v, out=vhat)
            else:
                vhat = v
            param.data.inplace_axpby(
                1.0 - hp.weight_decay_rate, -hp.eta,
                self.alpha_t * m / (numpy.sqrt(vhat) + hp.eps))
        else:
            m += (1 - hp.beta1) * (grad - m)
            v += (1 - hp.beta2) * (grad * grad - v)
            if hp.amsgrad:
                vhat = self.state['vhat']
                numpy.maximum(vhat, v, out=vhat)
            else:
                vhat = v
            param.data -= hp.eta * (
                self.alpha_t * m / (numpy.sqrt(vhat) + hp.eps) +
                hp.weight_decay_rate * param.data)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        if hp.amsgrad:
            if AdamRule._amsgrad_kernel is None:
                AdamRule._amsgrad_kernel = cuda.elementwise(
                    'T grad, T alpha_t, T one_minus_beta1, T one_minus_beta2, '
                    'T eps, T eta, T weight_decay_rate',
                    'T param, T m, T v, T vhat',
                    '''m += one_minus_beta1 * (grad - m);
                       v += one_minus_beta2 * (grad * grad - v);
                       vhat = max(vhat, v);
                       param -= eta * (alpha_t * m / (sqrt(vhat) + eps) +
                                       weight_decay_rate * param);''',
                    'adam')
            AdamRule._amsgrad_kernel(
                grad, self.alpha_t, 1 - hp.beta1,
                1 - hp.beta2, hp.eps,
                hp.eta, hp.weight_decay_rate,
                param.data, self.state['m'], self.state['v'],
                self.state['vhat'])
        else:
            if AdamRule._kernel is None:
                AdamRule._kernel = cuda.elementwise(
                    'T grad, T alpha_t, T one_minus_beta1, T one_minus_beta2, '
                    'T eps, T eta, T weight_decay_rate',
                    'T param, T m, T v',
                    '''m += one_minus_beta1 * (grad - m);
                       v += one_minus_beta2 * (grad * grad - v);
                       param -= eta * (alpha_t * m / (sqrt(v) + eps) +
                                       weight_decay_rate * param);''',
                    'adam')
            AdamRule._kernel(grad, self.alpha_t, 1 - hp.beta1,
                             1 - hp.beta2, hp.eps,
                             hp.eta, hp.weight_decay_rate,
                             param.data, self.state['m'], self.state['v'])

    @property
    def alpha_t(self):
        return _learning_rate(self.hyperparam, self.t)

    @property
    def lr(self):
        warnings.warn(
            'AdamRule.lr has been renamed to AdamRule.alpha_t. '
            'Use of AdamRule.lr is deprecated in Chainer v6.',
            DeprecationWarning)
        return self.alpha_t


class Adam(optimizer.GradientMethod):

    """Adam optimizer.

    See: `Adam: A Method for Stochastic Optimization \
          <https://arxiv.org/abs/1412.6980v8>`_

    Modified for proper weight decay (also called AdamW).
    AdamW introduces the additional parameters ``eta``
    and ``weight_decay_rate``, which can be used to properly scale the
    learning rate, and decouple the weight decay rate from ``alpha``,
    as shown in the below paper.

    Note that with the default values ``eta = 1`` and
    ``weight_decay_rate = 0``, this implementation is identical to
    the standard Adam method.

    See: `Fixing Weight Decay Regularization in Adam \
          <https://openreview.net/forum?id=rk6qdGgCZ>`_

    A flag ``amsgrad`` to use the AMSGrad variant of Adam from
    the paper: `On the Convergence of Adam and Beyond \
               <https://openreview.net/forum?id=ryQu7f-RZ>`_

    Args:
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.
        amsgrad (bool): Whether to use AMSGrad variant of Adam.

    """

    def __init__(self,
                 alpha=_default_hyperparam.alpha,
                 beta1=_default_hyperparam.beta1,
                 beta2=_default_hyperparam.beta2,
                 eps=_default_hyperparam.eps,
                 eta=_default_hyperparam.eta,
                 weight_decay_rate=_default_hyperparam.weight_decay_rate,
                 amsgrad=_default_hyperparam.amsgrad):
        super(Adam, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.eps = eps
        self.hyperparam.eta = eta
        self.hyperparam.weight_decay_rate = weight_decay_rate
        self.hyperparam.amsgrad = amsgrad

    alpha = optimizer.HyperparameterProxy('alpha')
    beta1 = optimizer.HyperparameterProxy('beta1')
    beta2 = optimizer.HyperparameterProxy('beta2')
    eps = optimizer.HyperparameterProxy('eps')
    eta = optimizer.HyperparameterProxy('eta')
    weight_decay_rate = optimizer.HyperparameterProxy('weight_decay_rate')
    amsgrad = optimizer.HyperparameterProxy('amsgrad')

    def create_update_rule(self):
        return AdamRule(self.hyperparam)

    @property
    def alpha_t(self):
        return _learning_rate(self.hyperparam, self.t)

    @property
    def lr(self):
        warnings.warn(
            'Adam.lr has been renamed to AdamRule.alpha_t. '
            'Use of Adam.lr is deprecated in Chainer v6.',
            DeprecationWarning)
        return self.alpha_t
