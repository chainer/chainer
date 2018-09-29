from __future__ import division
import math

import numpy

from chainer import optimizer
from chainer.optimizers import adam


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.alpha = 0.001
_default_hyperparam.beta1 = 0.9
_default_hyperparam.beta2 = 0.999
_default_hyperparam.beta3 = 0.999
_default_hyperparam.c = 10.0
_default_hyperparam.eps = 1e-8
_default_hyperparam.eta = 1.0
_default_hyperparam.f_star = 0.0
_default_hyperparam.weight_decay_rate = 0
_default_hyperparam.amsgrad = False


def _learning_rate(hp, t, d_tilde):
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Eve optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    return (hp.alpha / d_tilde) * math.sqrt(fix2) / fix1


class EveRule(adam.AdamRule):

    """Update rule of Eve optimization algorithm.

    See: https://arxiv.org/abs/1611.01505v3

    Before calling :meth:`update`, :attr:`d_tilde` must be set.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.
        amsgrad (bool): Whether to use the AMSGrad variant of Eve.
    """

    d_tilde = None

    @property
    def lr(self):
        assert self.d_tilde is not None
        return _learning_rate(self.hyperparam, self.t, self.d_tilde)


class Eve(optimizer.GradientMethod):

    """Eve optimizer.

    See: https://arxiv.org/abs/1611.01505v3

    Args:
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        beta3 (float): Exponential decay rate of the objective-dependent
            coefficient of learning rate.
        c (float): Constant used to clip the objective-dependent coefficient.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        f_star (float): Minimum value that the loss function can take.
        weight_decay_rate (float): Weight decay rate.
        amsgrad (bool): Whether to use AMSGrad variant of Eve.

    """

    def __init__(self,
                 alpha=_default_hyperparam.alpha,
                 beta1=_default_hyperparam.beta1,
                 beta2=_default_hyperparam.beta2,
                 beta3=_default_hyperparam.beta3,
                 c=_default_hyperparam.c,
                 eps=_default_hyperparam.eps,
                 eta=_default_hyperparam.eta,
                 f_star=_default_hyperparam.f_star,
                 weight_decay_rate=_default_hyperparam.weight_decay_rate,
                 amsgrad=_default_hyperparam.amsgrad):
        super(Eve, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.beta3 = beta3
        self.hyperparam.c = c
        self.hyperparam.eps = eps
        self.hyperparam.eta = eta
        self.hyperparam.f_star = f_star
        self.hyperparam.weight_decay_rate = weight_decay_rate
        self.hyperparam.amsgrad = amsgrad

    alpha = optimizer.HyperparameterProxy('alpha')
    beta1 = optimizer.HyperparameterProxy('beta1')
    beta2 = optimizer.HyperparameterProxy('beta2')
    beta3 = optimizer.HyperparameterProxy('beta3')
    c = optimizer.HyperparameterProxy('c')
    eps = optimizer.HyperparameterProxy('eps')
    eta = optimizer.HyperparameterProxy('eta')
    f_star = optimizer.HyperparameterProxy('f_star')
    weight_decay_rate = optimizer.HyperparameterProxy('weight_decay_rate')
    amsgrad = optimizer.HyperparameterProxy('amsgrad')

    def setup(self, link):
        """Sets a target link and initializes the optimizer states.

        Given link is set to the :attr:`target` attribute. It also prepares the
        optimizer state dictionaries corresponding to all parameters in the
        link hierarchy. The existing states are discarded.

        Args:
            link (~chainer.Link): Target link object.

        Returns:
            The optimizer instance.

        .. note::
           As of v4.0.0, this function returns the optimizer instance itself
           so that you can instantiate and setup the optimizer in one line,
           e.g., ``optimizer = SomeOptimizer().setup(link)``.

        """
        super(Eve, self).setup(link)
        self.d_tilde = numpy.nan
        self.f = numpy.nan
        return self

    def create_update_rule(self):
        return EveRule(self.hyperparam)

    @property
    def lr(self):
        return _learning_rate(self.hyperparam, self.t, self.d_tilde)

    def update(self, lossfun, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        Because Eve uses loss values, `lossfun` is required unlike in the
        case of other optimizers.

        Args:
            lossfun (callable): Callable that returns a ~chainer.Variable to be
                minimized.
            *args, **kwds: Arguments passed to `lossfun`.

        """
        assert lossfun is not None, 'Eve requires lossfun to be specified'
        use_cleargrads = getattr(self, '_use_cleargrads', True)
        loss = lossfun(*args, **kwds)
        if use_cleargrads:
            self.target.cleargrads()
        else:
            self.target.zerograds()
        loss.backward(loss_scale=self._loss_scale)
        loss_value = float(loss.array)
        del loss

        self.reallocate_cleared_grads()

        self.call_hooks('pre')

        self.t += 1
        self._update_d_tilde_and_f(loss_value)
        for param in self.target.params():
            param.update_rule.d_tilde = self.d_tilde
            param.update()

        self.reallocate_cleared_grads()

        self.call_hooks('post')

    def serialize(self, serializer):
        """Serializes or deserializes the optimizer.

        It only saves or loads the following things:

        - Optimizer states
        - Global states (:attr:`t`, :attr:`epoch`, :attr:`d_tilde`, and
            :attr:`f`)

        **It does not saves nor loads the parameters of the target link.** They
        should be separately saved or loaded.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer or
                deserializer object.
        """
        super(Eve, self).serialize(serializer)
        self.d_tilde = serializer('d_tilde', self.d_tilde)
        self.f = serializer('f', self.f)

    def _update_d_tilde_and_f(self, loss):
        if self.t > 1:
            d = abs(loss - self.f) / (min(loss, self.f) - self.f_star)
            d_hat = numpy.clip(d, 1/self.c, self.c)
            self.d_tilde = self.beta3 * self.d_tilde + (1 - self.beta3) * d_hat
        else:
            self.d_tilde = 1
        self.f = loss
