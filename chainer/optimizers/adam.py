from __future__ import division
import math
import warnings

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types
import chainerx


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


def _get_intermediate_dtype(dtype):
    # Returns the dtype for intermediate calculation.
    # For float16 input, float32 is used.
    # Otherwise the same dtype as the parameter is used.
    if dtype == numpy.float16:
        return numpy.float32
    return dtype


def _inplace_axpby(x, a, b, y):
    # in-place axpby: x = a * x + b * y
    if isinstance(x, intel64.mdarray):
        x.inplace_axpby(a, b, y)
    else:
        x[...] = a * x + b * y


@cuda.fuse(kernel_name='adam')
def _adam_impl(
        grad, alpha_t, one_minus_beta1, one_minus_beta2, eps,
        eta, one_minus_weight_decay_rate, param, m, v, vhat):
    xp = cuda.get_array_module(grad)
    dtype = _get_intermediate_dtype(param.dtype.type)
    grad = grad.astype(dtype, copy=False)

    # m += (1 - beta1) * (grad - m)
    _inplace_axpby(m, 1.0, one_minus_beta1, grad - m)
    # v += (1 - beta2) * (grad * grad - v)
    _inplace_axpby(v, 1.0, one_minus_beta2, grad * grad - v)

    if vhat is None:
        vhat = v
    else:
        xp.maximum(vhat, v, out=vhat)
    vhat = vhat.astype(dtype, copy=False)

    # param -= eta * (alpha_t * m / (sqrt(vhat) + eps)
    #                 - weight_decay_rate * param)
    _inplace_axpby(
        param, one_minus_weight_decay_rate, -eta,
        alpha_t * m / (xp.sqrt(vhat) + eps))


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

    def _check_eps(self, interm_dtype):
        # Checks that the eps does not underflow.
        hp = self.hyperparam
        eps = interm_dtype(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    interm_dtype.name, hp.eps))
        # Note that the converted `eps` (numpy scalar) is discarded here and
        # the original `hp.eps` is used in calculation, because Python
        # scalars are faster in cupy elementwise kernels.

    def update_core(self, param):
        device = param.device
        with chainer.using_device(device):
            if device.xp is chainerx:
                self.update_core_chainerx(param)
                return

            grad = param.grad
            if grad is None:
                return
            hp = self.hyperparam
            dtype = _get_intermediate_dtype(param.dtype.type)
            self._check_eps(dtype)

            vhat = self.state['vhat'] if self.hyperparam.amsgrad else None
            _adam_impl(grad, self.alpha_t, 1 - hp.beta1, 1 - hp.beta2, hp.eps,
                       hp.eta, 1 - hp.weight_decay_rate, param.data,
                       self.state['m'], self.state['v'], vhat)

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
