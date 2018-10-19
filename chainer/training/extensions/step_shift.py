from __future__ import division

import numpy

from chainer.training import extension


class StepShift(extension.Extension):

    """Trainer extension to shift an optimizer attribute in "steps".

    This extension multiplies the specified attribute of the optimizer in
    "steps". The typical use case is to scale the attribute at every ``k``\\ th
    iteration.

    For example, suppose that this extension is invoked at every iteration,
    then given ``k``, a multiplier ``gamma`` and an initial value
    ``init``, the optimizer attribute is set to
    ``init * gamma ^ (floor(i / k))``, where ``i`` represents the index of the
    current iteration.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        gamma (float): The multiplier.
        step (int): The interval for the multiplication, i.e., ``k``.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.

    """

    def __init__(self, attr, gamma, step, init=None, target=None,
                 optimizer=None):
        self._attr = attr
        self._gamma = gamma
        self._step = step
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self._init
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._init * self._gamma ** numpy.floor(self._t / self._step)
        if self._target is not None:
            if self._gamma > 1:
                # almost same as value = min(value, self._target), but this
                # line supports negative values, too
                if value / self._target > 1:
                    value = self._target
            else:
                # ditto
                if value / self._target < 1:
                    value = self._target
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
