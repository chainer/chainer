from __future__ import division

from chainer.training import extension


class WarmUp(extension.Extension):

    def __init__(self, attr, warmup_start_lr, base_lr, warmup_iter,
                 optimizer=None):
        self._attr = attr
        self._base_lr = base_lr
        self._warmup_iter = warmup_iter
        self._warmup_start_lr = warmup_start_lr
        self._optimizer = optimizer
        self._t = 0

    def initialize(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._warmup_start_lr is None:
            self._warmup_start_lr = getattr(optimizer, self._attr)
        else:
            setattr(optimizer, self._attr, self._warmup_start_lr)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        value = (self._t * self._base_lr + (self._warmup_iter - self._t)
                 * self._warmup_start_lr) / self._warmup_iter
        value = round(value, 6)
        if self._t <= self._warmup_iter:
            setattr(optimizer, self._attr, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
