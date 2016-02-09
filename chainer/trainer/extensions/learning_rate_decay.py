from chainer.trainer import extension


class LearningRateDecay(extension.Extension):

    """Trainer extension to periodically decay the learning rate.

    TODO(beam2d): document it.

    """
    default_trigger = 1, 'epoch'
    invoke_before_training = True

    def __init__(self, decay_rate, initial_lr=None, key='lr'):
        self._decay_rate = decay_rate
        self._initial_lr = initial_lr
        self._key = key
        self._decay_t = 0

    def __call__(self, trainer, **kwargs):
        lr = self._initial_lr
        if lr is None:
            lr = trainer.optimizer.lr
        lr *= self._decay_rate ** self._decay_t
        setattr(trainer.optimizer, self._key, lr)
        self._decay_t += 1

    def serialize(self, serializer):
        self._decay_t = serializer('_decay_t', self._decay_t)
