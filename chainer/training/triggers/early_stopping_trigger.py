from chainer.training import util


def greater(current_val, best_val):
    return current_val > best_val


def less(current_val, best_val):
    return current_val < best_val


class EarlyStoppingTrigger(object):
    """Trigger invoked when specific value continue to be worse.

    Args:
        monitor (str) : the metric you want to monitor
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
        patients (int) : the value to patient
        mode (str) : max, min, or auto. using them to determine the _compare
        verbose (bool) : flag for debug mode
        max_epoch (int) : upper bound of the number of training loops
    """

    def __init__(self, trigger=(1, 'epoch'), monitor='main/loss',
                 patients=3, mode='auto', verbose=False, max_epoch=100):

        self.count = 0
        self.patients = patients
        self.monitor = monitor
        self.verbose = verbose
        self.max_epoch = max_epoch
        self._interval_trigger = util.get_trigger(trigger)

        if mode == 'max':
            self._compare = greater

        elif mode == 'min':
            self._compare = less

        else:
            if 'accuracy' in monitor:
                self._compare = greater

            else:
                self._compare = less

        if self._compare == greater:
            if verbose:
                print('early stopping: operator is greater')
            self.best = -1 * (1 << 50)

        else:
            if verbose:
                print('early stopping: operator is less')
            self.best = 1 << 50

    def __call__(self, trainer):
        """Decides whether the training loop should be stopped.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.

        Returns:
            bool: ``True`` if the training loop should be stopped.
        """

        observation = trainer.observation

        if trainer.updater.epoch >= self.max_epoch:
            return True

        if self.monitor not in observation.keys():
            return False

        if not self._interval_trigger(trainer):
            return False

        current_val = float(observation[self.monitor].data)

        if self.verbose:
            print('current count: {}'.format(self.count))
            print('best: {}, current_val: {}'.format(self.best, current_val))

        if self._compare(current_val, self.best):
            self.best = current_val
            self.count = 0

        else:
            self.count += 1

        if self._stop_condition():
            return True

        return False

    def _stop_condition(self):
        if self.verbose:
            print('{} > {}'.format(self.count, self.patients))
        return self.count > self.patients
