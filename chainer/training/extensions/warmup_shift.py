from __future__ import division

from chainer.training import extension


class WarmupShift(extension.Extension):

    """Trainer extension to change an optimizer attribute evenly at the
    begining of one training.

    For example, suppose that this extension is called at every iteration,
    and warmup_start_lr = x , init = y, warmup_iter = t.
    Then this extension will set the corresponding attribute to from
    ``x`` to ``y`` evenly in first ``t`` iterations.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        warmup_start_lr (float): the value of the attr at the begining
            of one training.
        init (float): the value of the attr after warm up iterations.
        warmup_iter (int): the number of the iterations in which the
            attr changes from ``warmup_start_lr`` to ``init``.
        optimizer (~chainer.Optimizer): Target optimizer object.
            If it is None, the main optimizer of the trainer is used.

    """

    def __init__(self, attr, warmup_start_lr, init, warmup_iter,
                 optimizer=None):
        self._attr = attr
        self._init = init
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
        value = (self._t * self._init + (self._warmup_iter - self._t)
                 * self._warmup_start_lr) / self._warmup_iter
        if self._t <= self._warmup_iter:
            setattr(optimizer, self._attr, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
