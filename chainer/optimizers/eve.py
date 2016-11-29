import math

import numpy

from chainer import cuda
from chainer import optimizer


class Eve(optimizer.GradientMethod):

    """Eve optimization algorithm.

    See: https://arxiv.org/abs/1611.01505

    You must speficify ``lossfun`` when you call its ``update`` method so that
    it can utilize loss values in optimization.
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.999,
                 eps=1e-8, lower_threshold=0.1, upper_threshold=10):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['m'] = xp.zeros_like(param.data)
            state['v'] = xp.zeros_like(param.data)
            state['d'] = xp.ones(1, dtype=param.data.dtype)
            state['f'] = xp.zeros(1, dtype=param.data.dtype)

    def _update_d_and_f(self, state):
        d, f = state['d'], state['f']
        if self.t > 1:
            old_f = float(cuda.to_cpu(state['f']))
            if self.loss > old_f:
                delta = self.lower_threshold + 1.
                Delta = self.upper_threshold + 1.
            else:
                delta = 1. / (self.upper_threshold + 1.)
                Delta = 1. / (self.lower_threshold + 1.)
            c = min(max(delta, self.loss / old_f), Delta)
            new_f = c * old_f
            r = abs(new_f - old_f) / min(new_f, old_f)
            d += (1 - self.beta3) * (r - d)
            f[:] = new_f
        else:
            f[:] = self.loss

    def update_one_cpu(self, param, state):
        m, v, d = state['m'], state['v'], state['d']
        grad = param.grad

        self._update_d_and_f(state)
        m += (1. - self.beta1) * (grad - m)
        v += (1. - self.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (d * numpy.sqrt(v) + self.eps)

    def update_one_gpu(self, param, state):
        self._update_d_and_f(state)
        cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps, T d',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (d * sqrt(v) + eps);''',
            'eve')(param.grad, self.lr, 1 - self.beta1, 1 - self.beta2,
                   self.eps, float(state['d']), param.data, state['m'],
                   state['v'])

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * math.sqrt(fix2) / fix1

    def update(self, lossfun=None, *args, **kwds):
        # Overwrites GradientMethod.update in order to get loss values
        if lossfun is None:
            raise RuntimeError('Eve.update requires lossfun to be specified')
        loss_var = lossfun(*args, **kwds)
        self.loss = float(loss_var.data)
        super(Eve, self).update(lossfun=lambda: loss_var)
