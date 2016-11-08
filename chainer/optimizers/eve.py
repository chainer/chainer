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
            if self.loss < old_f:
                delta = self.lower_threshold + 1.
                Delta = self.upper_threshold + 1.
            else:
                delta = 1. / (self.upper_threshold + 1.)
                Delta = 1. / (self.lower_threshold + 1.)
            c = min(max(delta, self.loss / old_f), Delta)
            new_f = c * self.loss
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

        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            self.loss = float(loss.data)
            del loss
        elif len(kwds) == 1:
            for name in kwds:
                self.loss = float(kwds[name].data)
        else:
            raise RuntimeError('Eve algorithm require lossfun or loss variable in order to utilize past loss values.')

        # TODO(unno): Some optimizers can skip this process if they does not
        # affect to a parameter when its gradient is zero.
        for name, param in self.target.namedparams():
            if param.grad is None:
                with cuda.get_device(param.data):
                    xp = cuda.get_array_module(param.data)
                    param.grad = xp.zeros_like(param.data)

        self.call_hooks()
        self.prepare()

        self.t += 1
        states = self._states
        for name, param in self.target.namedparams():
            with cuda.get_device(param.data):
                self.update_one(param, states[name])
