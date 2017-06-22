from __future__ import division

import numpy as np

from chainer.training import extension


class LinearShift(extension.Extension):

    """Trainer extension to change an optimizer attribute linearly.

    This extension changes an optimizer attribute from the first value to the
    last value linearly within a specified duration. The typical use case is
    warming up of the momentum coefficient.

    For example, suppose that this extension is called at every iteration, and
    ``value_range == (x, y)`` and ``time_range == (i, j)``. Then, this
    extension keeps the attribute to be ``x`` up to the ``i``-th iteration,
    linearly shifts the value to ``y`` by the ``j``-th iteration, and then
    keeps the value to be ``y`` after the ``j``-th iteration.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        value_range (tuple of float): The first and the last values of the
            attribute.
        time_range (tuple of ints): The first and last counts of calls in which
            the attribute is adjusted.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.

    """

    def __init__(self, attr, value_range, time_range, optimizer=None):
        self._attr = attr
        self._value_range = value_range
        self._time_range = time_range
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self._compute_next_value()
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._compute_next_value()
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _compute_next_value(self):
        t1, t2 = self._time_range
        v1, v2 = self._value_range

        if self._t <= t1:
            return v1
        elif self._t >= t2:
            return v2
        rate = (self._t - t1) / (t2 - t1)
        return v1 + rate * (v2 - v1)

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
