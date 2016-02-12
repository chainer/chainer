from chainer.trainer import extension


class LinearShift(extension.Extension):

    """Shifts an optimizer attribute linearly within given duration.

    TODO(beam2d): document it.

    """
    invoke_before_training = True

    def __init__(self, attr, value_range, time_range, optimizer=None):
        self._attr = attr
        self._value_range = value_range
        self._time_range = time_range
        self._optimizer = optimizer
        self._t = 0

    def __call__(self, trainer, **kwargs):
        optimizer = self._optimizer or trainer.optimizer
        t1, t2 = self._time_range
        v1, v2 = self._value_range

        if self._t <= t1:
            value = v1
        elif self._t >= t2:
            value = v2
        else:
            rate = float(t2 - t1) / (self._t - t1)
            value = v1 + rate * (v2 - v1)
        setattr(optimizer, self._attr, value)

        self._t += 1

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
