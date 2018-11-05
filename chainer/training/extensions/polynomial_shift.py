from __future__ import division

import numpy

from chainer.training import extension


class PolynomialShift(extension.Extension):

    """Trainer extension to polynomially shift an optimizer attribute.

    This extension polynomially decreases the specified attribute of the
    optimizer. The typical use case is a polynomial decay of the
    learning rate at each iteration.

    For example, suppose that this extension is invoke at every iteration.
    Then this extension will set the corresponding attribute to
    ``init_value * (1 - i / max_iter) ^ rate`` at the ``i``-th iteration, where
    the ``max_iter`` is the number of iterations to be running.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Exponent of polynomial shift.
        max_count (int): Number of this extension to be invoked.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """
    invoke_before_training = True

    def __init__(self, attr, rate, max_count, init=None, target=None,
                 optimizer=None):
        self._attr = attr
        self._rate = rate
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._max_count = max_count
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

        decay = max(1 - self._t / self._max_count, 0)
        value = self._init * decay ** self._rate

        if self._target is not None:
            if self._rate > 0:
                # almost same as value = min(value, self._target), but this
                # line supports negative values, too
                if self._target / value > 1:
                    value = self._target
            else:
                # ditto
                if self._target / value < 1:
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
