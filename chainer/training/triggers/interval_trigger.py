import warnings


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
        if unit not in ('epoch', 'iteration'):
            raise ValueError(
                'Trigger unit must be either \'epoch\' or \'iteration\'.')

        self.period = period
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

            # if previous_epoch_detail is invalid value,
            # use the value of updater.
            if previous_epoch_detail < 0:
                previous_epoch_detail = updater.previous_epoch_detail

            # count is kept for backward compatibility
            self.count = epoch_detail // self.period

            fire = previous_epoch_detail // self.period != \
                epoch_detail // self.period
        else:
            iteration = updater.iteration
            previous_iteration = self._previous_iteration

            # if previous_iteration is invalid value,
            # guess it from current iteration.
            if previous_iteration < 0:
                previous_iteration = iteration - 1

            fire = previous_iteration // self.period != \
                iteration // self.period

        # save current values
        self._previous_iteration = updater.iteration
        if hasattr(updater, 'epoch_detail'):
            self._previous_epoch_detail = updater.epoch_detail

        return fire

    def serialize(self, serializer):
        try:
            self._previous_iteration = serializer(
                'previous_iteration', self._previous_iteration)
        except KeyError:
            warnings.warn(
                'The previous value of iteration is not saved. '
                'IntervalTrigger guesses it using current iteration. '
                'If this trigger is not called at every iteration, '
                'it may not work correctly.')
            # set a negative value for invalid
            self._previous_iteration = -1

        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            warnings.warn(
                'The previous value of epoch_detail is not saved. '
                'IntervalTrigger uses the value of '
                'trainer.updater.previous_epoch_detail. '
                'If this trigger is not called at every iteration, '
                'it may not work correctly.')
            # set a negative value for invalid
            self._previous_epoch_detail = -1.

    def get_training_length(self):
        return (self.period, self.unit)
