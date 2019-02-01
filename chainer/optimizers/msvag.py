from __future__ import division

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class MSVAGHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of M-SVAG.

        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        beta = None  # type: float
        eta = None  # type: float
        weight_decay_rate = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: MSVAGHyperparameter # NOQA
_default_hyperparam.lr = 0.1
_default_hyperparam.beta = 0.9
_default_hyperparam.eta = 1.0
_default_hyperparam.weight_decay_rate = 0


class MSVAGRule(optimizer.UpdateRule):

    """Update rule of the M-SVAG optimization algorithm.

    See: `Dissecting Adam: The Sign, Magnitude and Variance of Stochastic \
          Gradients <https://arxiv.org/abs/1705.07774>`_

    Modified for proper weight decay.

    See: `Fixing Weight Decay Regularization in Adam \
          <https://openreview.net/forum?id=rk6qdGgCZ>`_

    See :class:`~chainer.optimizers.MSVAG` for the default values
    of the hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        beta (float): Exponential decay rate of the first and second order
                      moment.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.

    """

    def __init__(self, parent_hyperparam=None,
                 lr=None, beta=None,
                 eta=None, weight_decay_rate=None):
        super(MSVAGRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if beta is not None:
            self.hyperparam.beta = beta
        if eta is not None:
            self.hyperparam.eta = eta
        if weight_decay_rate is not None:
            self.hyperparam.weight_decay_rate = weight_decay_rate

        self.beta_power = self.hyperparam.beta

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        m, v = self.state['m'], self.state['v']

        rho = (((1.0 - hp.beta) ** 2) * (1.0 - self.beta_power ** 2) /
               (((1.0 - self.beta_power) ** 2) * (1.0 - hp.beta ** 2)))
        rho = min(rho, 0.9999)

        m += (1 - hp.beta) * (grad - m)
        v += (1 - hp.beta) * (grad * grad - v)

        mt = m / (1 - self.beta_power)
        vt = v / (1 - self.beta_power)

        mt2 = mt ** 2
        s = (vt - mt2) / (1 - rho)

        factor = numpy.clip(mt2 / (mt2 + rho * s), 0, 1)
        if isinstance(factor, numpy.ndarray):
            factor[numpy.isnan(factor)] = 0
        else:
            if numpy.isnan(factor):
                factor = 0

        param.data -= hp.eta * (hp.lr * mt * factor +
                                hp.weight_decay_rate * param.data)

        self.beta_power *= hp.beta

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        hp = self.hyperparam

        rho = (((1.0 - hp.beta) ** 2) * (1.0 - self.beta_power ** 2) /
               (((1.0 - self.beta_power) ** 2) * (1.0 - hp.beta ** 2)))
        rho = min(rho, 0.9999)

        cuda.elementwise(
            'T grad, T lr, T one_minus_beta, T eta, \
             T weight_decay_rate, T beta_power, T rho',
            'T param, T m, T v',
            '''m += one_minus_beta * (grad - m);
               v += one_minus_beta * (grad * grad - v);

               T mt = m / (1.0 - beta_power);
               T vt = v / (1.0 - beta_power);

               T mt2 = mt*mt;
               T s = (vt - mt2) / (1.0 - rho);

               T factor;
               if (m == 0 && v == 0)
                   factor = 0.0;
               else
                   factor = min(1.0, max(0.0, mt2 / (mt2 + rho * s)));

               param -= eta * (lr * mt * factor +
                               weight_decay_rate * param);''',
            'msvag')(grad, hp.lr, 1 - hp.beta,
                     hp.eta, hp.weight_decay_rate,
                     self.beta_power, rho,
                     param.data, self.state['m'], self.state['v'])

        self.beta_power *= hp.beta


class MSVAG(optimizer.GradientMethod):

    """M-SVAG optimizer.

    See: `Dissecting Adam: The Sign, Magnitude and Variance of Stochastic \
          Gradients <https://arxiv.org/abs/1705.07774>`_

    Modified for proper weight decay (also called AdamW).
    AdamW introduces the additional parameters ``eta``
    and ``weight_decay_rate``, which can be used to properly scale the
    learning rate, and decouple the weight decay rate from ``alpha``,
    as shown in the below paper.

    See: `Fixing Weight Decay Regularization in Adam \
          <https://openreview.net/forum?id=rk6qdGgCZ>`_

    Args:
        lr (float): Learning rate.
        beta (float): Exponential decay rate of the first and second order
                      moment.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.

    """

    def __init__(self,
                 lr=_default_hyperparam.lr,
                 beta=_default_hyperparam.beta,
                 eta=_default_hyperparam.eta,
                 weight_decay_rate=_default_hyperparam.weight_decay_rate):
        super(MSVAG, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.beta = beta
        self.hyperparam.eta = eta
        self.hyperparam.weight_decay_rate = weight_decay_rate

    lr = optimizer.HyperparameterProxy('lr')
    beta = optimizer.HyperparameterProxy('beta')
    eta = optimizer.HyperparameterProxy('eta')
    weight_decay_rate = optimizer.HyperparameterProxy('weight_decay_rate')

    def create_update_rule(self):
        return MSVAGRule(self.hyperparam)
