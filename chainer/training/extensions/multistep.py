from __future__ import division

from chainer.training import extension


class Multistep(extension.Extension):

    def __init__(self, attr, base_lr, gamma, step_value, optimizer=None):
        self._attr = attr
        self._base_lr = base_lr
        self._gamma = gamma
        self._step_value = step_value
        self._stepvalue_size = len(step_value)
        self._optimizer = optimizer
        self._current_step = 0
        self._t = 0

    def initialize(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._base_lr is None:
            self._base_lr = getattr(optimizer, self._attr)
        else:
            setattr(optimizer, self._attr, self._base_lr)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._current_step < self._stepvalue_size and \
           self._t >= self._step_value[self._current_step]:
            self._current_step += 1
        value = self._base_lr * pow(self._gamma, self._current_step)
        value = round(value, 6)
        setattr(optimizer, self._attr, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
