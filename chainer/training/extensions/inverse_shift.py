from __future__ import division

import numpy

from chainer.training import extension


class InverseShift(extension.Extension):

    """Trainer extension to shift an optimizer attribute.

    The new value is computed according to the fomula below:
    new_attr = init_attr * (1 + gamma * iter) ^ (- power), which is compatible
    to the ``inv`` learning rate policy in Caffe.

    The typical use is to decrease the learning rate during the training.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        gamma (float): Parameter used to compute the new value. Refer to the
            fomula above. Note that gamma is assumed to be nonegative.
        power (float): Parameter used to compute the new value. Refer to the
            fomula above.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.
    """

    def __init__(self, attr, gamma, power,
                 init=None, target=None, optimizer=None):
        self._attr = attr
        if gamma < 0:
            raise ValueError('InverseShift does not support negative gamma')
        self._gamma = gamma
        self._power = power
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

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = self._init * (1 + self._gamma * self._t) ** (-self._power)
        if self._target is not None:
            if self._power < 0:
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
