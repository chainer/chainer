import warnings


class ManualScheduleTrigger(object):

    """Trigger invoked at specified point(s) of iterations or epochs.

    This trigger accepts iterations or epochs indicated by given point(s).
    There are two ways to specify the point(s): iteration and epoch.
    ``iteration`` means the number of updates, while ``epoch`` means the number
    of sweeps over the training dataset. Fractional values are allowed
    if the point is a number of epochs; the trigger uses the ``iteration``
    and ``epoch_detail`` attributes defined by the updater.

    Args:
        points (int, float, or list of int or float): time of the trigger.
            Must be an integer or list of integer if unit is ``'iteration'``.
        unit (str): Unit of the time specified by ``points``. It must be
            either ``'iteration'`` or ``'epoch'``.

    Attributes:
        finished (bool): Flag that indicates whether or not this trigger will
        fire in the future. This flag is used to determine if the extension
        should be initialized after resume.

    """

    def __init__(self, points, unit):
        if unit not in ('epoch', 'iteration'):
            raise ValueError(
                'Trigger unit must be either \'epoch\' or \'iteration\'.')

        self.points = (points if isinstance(points, list) else [points])
        self.unit = unit
        self.finished = False

        self._previous_iteration = 0
        self._previous_epoch_detail = 0.

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

            fire = any(
                previous_epoch_detail < p <= epoch_detail
                for p in self.points)

            if hasattr(self, '_finished_is_tmp'):
                del self._finished_is_tmp
                if epoch_detail >= max(self.points):
                    self.finished = True
            if fire and epoch_detail >= max(self.points):
                self.finished = True
        else:
            iteration = updater.iteration
            previous_iteration = self._previous_iteration

            # if previous_iteration is invalid value,
            # guess it from current iteration.
            if previous_iteration < 0:
                previous_iteration = iteration - 1

            fire = any(
                previous_iteration < p <= iteration
                for p in self.points)

            if hasattr(self, '_finished_is_tmp'):
                del self._finished_is_tmp
                if iteration >= max(self.points):
                    self.finished = True
            if fire and iteration >= max(self.points):
                self.finished = True

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
                'ManualScheduleTrigger guesses it using current iteration. '
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
                'ManualScheduleTrigger uses the value of '
                'trainer.updater.previous_epoch_detail. '
                'If this trigger is not called at every iteration, '
                'it may not work correctly.')
            # set a negative value for invalid
            self._previous_epoch_detail = -1.

        try:
            self.finished = serializer('finished', self.finished)
        except KeyError:
            warnings.warn(
                'The flag of finished is not saved. '
                'ManualScheduleTrigger set the flag to `False` to force '
                'initialization and reset in next `__call__`.')
            # set False to force initialization.
            self.finished = False
            self._finished_is_tmp = True
