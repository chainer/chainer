import operator
import warnings

from chainer import reporter
from chainer.training import util
from chainer.utils import argument


class EarlyStoppingTrigger(object):
    """__init__(\
        self, check_trigger=(1, 'epoch'), monitor='main/loss', \
        patience=3, mode='auto', verbose=False, \
        max_trigger=(100, 'epoch'))

    Trigger for Early Stopping

    It can be used as a stop trigger of :class:`~chainer.training.Trainer`
    to realize *early stopping* technique.

    This trigger works as follows.
    Within each *check interval* defined by the ``check_trigger`` argument,
    it monitors and accumulates the reported value at each iteration.
    At the end of each interval, it computes the mean of the accumulated
    values and compares it to the previous ones to maintain the *best* value.
    When it finds that the best value is not updated
    for some periods (defined by ``patience``), this trigger fires.

    Args:
        monitor (str) : The metric you want to monitor
        check_trigger: Trigger that decides the comparison
            interval between current best value and new value.
            This must be a tuple in the form of ``<int>,
            'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
        patience (int) : Counts to let the trigger be patient.
            The trigger will not fire until the condition is met
            for successive ``patience`` checks.
        mode (str) : ``'max'``, ``'min'``, or ``'auto'``.
            It is used to determine how to compare the monitored values.
        verbose (bool) : Enable verbose output.
            If verbose is true, you can get more information
        max_trigger: Upper bound of the number of training loops

    .. note::
       ``patients`` is also available as an alias of ``patience`` for
       historical reason.
    """

    def __init__(self, check_trigger=(1, 'epoch'), monitor='main/loss',
                 patience=None, mode='auto', verbose=False,
                 max_trigger=(100, 'epoch'), **kwargs):

        # `patients` as an alias of `patience`
        patients, = argument.parse_kwargs(kwargs, ('patients', None))
        if patients is None:
            if patience is None:
                patience = 3
            else:
                pass
        else:
            if patience is None:
                patience = patients
            else:
                raise TypeError(
                    'Both \'patience\' and \'patients\' arguments are '
                    'specified. \'patients\' is an alias of the former. '
                    'Specify only \'patience\'.')

        self.count = 0
        self.patience = patience
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
        return self.count >= self.patience

    def _init_summary(self):
        self._summary = reporter.DictSummary()

    def get_training_length(self):
        return self._max_trigger.get_training_length()
