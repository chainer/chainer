from __future__ import division

from chainer.training import extension


class Poly(extension.Extension):

    """Trainer extension to poly shift an optimizer attribute.

    This extension poly increases or decreases the specified attribute
    of the optimizer. The typical use case is an exponential decay of the
    learning rate.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        (rate, max_itr) (float, int): Rate of the exponential shift.
             This value is multiplied to the attribute at each call.
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

    def __init__(self, attr, param, init=None, target=None, optimizer=None):
        self._attr = attr
        if param[0] < 0:
            raise ValueError('ExponentialShift does not support negative rate')
        self._rate = param[0]
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._max_iter = param[1]

    def __call__(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._init is None:
            self._init = getattr(optimizer, self._attr)
        # value = self._init * (self._rate ** self._t)
        decay = max(1 - self._t / self._max_iter, 0)
        value = self._init * decay ** self._rate
        if self._target is not None:
            if self._rate > 1:
                # almost same as value = min(value, self._target), but this
                # line supports negative values, too
                if value / self._target > 1:
                    value = self._target
            else:
                # ditto
                if value / self._target < 1:
                    value = self._target
        setattr(optimizer, self._attr, value)
        self._t += 1

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
