from chainer.trainer import extension


class ExponentialDecay(extension.Extension):

    """Trainer extension to exponentially decay an optimizer attribute.

    TODO(beam2d): document it.

    """
    invoke_before_training = True

    def __init__(self, attr, decay_rate, init=None, minimum=None,
                 optimizer=None):
        self._attr = attr
        self._decay_rate = decay_rate
        self._init = init
        self._minimum = minimum
        self._optimizer = optimizer
        self._t = 0

    def __call__(self, trainer):
        optimizer = self._optimizer or trainer.optimizer
        if self._init is None:
            self._init = getattr(optimizer, self._attr)
        value = max(self._init * (self._decay_rate ** self._t),
                    self._minimum)
        setattr(optimizer, self._attr, value)
        self._t += 1

    def serialize(self, serializer):
        self._decay_t = serializer('_t', self._t)
