class IntervalTrigger(object):

    """Trigger based on a fixed interval.

    This trigger accepts iterations divided by a given interval. There are two
    ways to specify the interval: iteration and epoch. Here `iteration` means
    the number of updates, while `epoch` means the number of sweeps over the
    training dataset.

    Args:
        period (int): Length of the interval.
        unit (str): Unit of the length specified by ``period``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """
    def __init__(self, period, unit):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer (Trainer): Trainer object that currently runs.

        Returns:
            bool: True if the corresponding extension should be invoked in this
                iteration.

        """
        if self.unit == 'epoch':
            return trainer.new_epoch and trainer.epoch % self.period == 0
        else:
            return trainer.t % self.period == 0
