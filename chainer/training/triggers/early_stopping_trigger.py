from chainer import reporter
from chainer.training import util


import chainer
import operator
import warnings


class EarlyStoppingTrigger(object):
    """Trigger invoked when specific value continue to be worse.

    Args:
        monitor (str) : The metric you want to monitor
        check_trigger: Trigger that decides the comparison
            interval between current best value and new value.
            This must be a tuple in the form of ``<int>,
            'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
        patients (int) : Counts to let the trigger be patient.
            The trigger will not fire until the condition is met
            for successive ``patient`` checks.
        mode (str) : ``'max'``, ``'min'``, or ``'auto'``.
            It is used to determine how to compare the monitored values.
        verbose (bool) : Enable verbose output.
            If verbose is true, you can get more information
        max_trigger (int) : Upper bound of the number of training loops
    """

    def __init__(self, check_trigger=(1, 'epoch'), monitor='main/loss',
                 patients=3, mode='auto', verbose=False,
                 max_trigger=(100, 'epoch')):

        self.count = 0
        self.patients = patients
        self.monitor = monitor
        self.verbose = verbose
        self.already_warning = False
        self._max_trigger = util.get_trigger(max_trigger)
        self._interval_trigger = util.get_trigger(check_trigger)

        self._init_summary()

        if mode == 'max':
            self._compare = operator.gt

        elif mode == 'min':
            self._compare = operator.lt

        else:
            if 'accuracy' in monitor:
                self._compare = operator.gt

            else:
                self._compare = operator.lt

        if self._compare == operator.gt:
            if verbose:
                print('early stopping: operator is greater')
            self.best = float('-inf')

        else:
            if verbose:
                print('early stopping: operator is less')
            self.best = float('inf')

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

        summary = self._summary

        if self.monitor in observation:
            summary.add({self.monitor: observation[self.monitor]})

        if self._max_trigger(trainer):
            return True

        if not self._interval_trigger(trainer):
            return False

        if self.monitor not in observation.keys():
            warnings.warn('{} is not in observation'.format(self.monitor))
            return False

        stat = self._summary.compute_mean()
        current_val = stat[self.monitor]
        self._init_summary()

        if chainer.is_debug():
            print('current count: {}'.format(self.count))
            print('best: {}, current_val: {}'.format(self.best, current_val))

        if self._compare(current_val, self.best):
            self.best = current_val
            self.count = 0

        else:
            self.count += 1

        if self._stop_condition():
            if self.verbose:
                print('Epoch {}: early stopping'.format(trainer.updater.epoch))
            return True

        return False

    def _stop_condition(self):
        if chainer.is_debug():
            print('{} >= {}'.format(self.count, self.patients))
        return self.count >= self.patients

    def _init_summary(self):
        self._summary = reporter.DictSummary()
