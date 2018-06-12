from __future__ import division

import numpy

from chainer.training import extension


class StepShift(extension.Extension):

    """Trainer extension to multiply an optimizer attribute by a fixed value
    for every ``k`` iterations.

    For example, given ``k``, a multiplier ``gamma`` and an initial value
    ``init``, the optimizer attribute is set to
    ``init * gamma ^ (floor(iter / k))``.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        init (float): The initial value of the attribute.
        gamma (float): The multiplier.
        step (int): The interval for the multiplication, i.e., ``k``.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.

    """

    def __init__(self, attr, init, gamma, step, optimizer=None):
        self._attr = attr
        self._init = init
        self._gamma = gamma
        self._step = step
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self._init
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._init * self._gamma ** numpy.floor(self._t / self._step)
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = numpy.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
