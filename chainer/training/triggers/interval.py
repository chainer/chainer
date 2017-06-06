class IntervalTrigger(object):

    """Trigger based on a fixed interval.

    This trigger accepts iterations divided by a given interval. There are two
    ways to specify the interval: per iterations and epochs. `Iteration` means
    the number of updates, while `epoch` means the number of sweeps over the
    training dataset. Fractional values are allowed if the interval is a
    number of epochs; the trigger uses the `iteration` and `epoch_detail`
    attributes defined by the updater.

    For the description of triggers, see :func:`~chainer.training.get_trigger`.

    Args:
        period (int or float): Length of the interval. Must be an integer if
            unit is ``'iteration'``.
        unit (str): Unit of the length specified by ``period``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """

    def __init__(self, period, unit):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit

        self._previous_iteration = 0
        self._previous_epoch_detail = 0.

        # count is kept for backward compatibility
        self.count = 0

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer (Trainer): Trainer object that this trigger is associated
                with. The updater associated with this trainer is used to
                determine if the trigger should fire.

        Returns:
            bool: True if the corresponding extension should be invoked in this
                iteration.

        """
        updater = trainer.updater
        if self.unit == 'epoch':
            epoch_detail = updater.epoch_detail
            previous_epoch_detail = self._previous_epoch_detail

            self._previous_epoch_detail = epoch_detail

            # count is kept for backward compatibility
            self.count = epoch_detail // self.period

            return previous_epoch_detail // self.period != \
                epoch_detail // self.period
        else:
            iteration = updater.iteration
            previous_iteration = self._previous_iteration

            self._previous_iteration = iteration

            return previous_iteration // self.period != \
                iteration // self.period

    def serialize(self, serializer):
        self._previous_iteration = serializer(
            'previous_iteration', self._previous_iteration)
        self._previous_epoch_detail = serializer(
            'previous_epoch_detail', self._previous_epoch_detail)
