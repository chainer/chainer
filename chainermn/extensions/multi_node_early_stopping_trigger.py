from chainer.training.triggers import EarlyStoppingTrigger
from chainermn.extensions import ObservationAggregator


class MultiNodeEarlyStoppingTrigger(object):
    """__init__(\
        self, comm, check_trigger=(1, 'epoch'), monitor='main/loss', \
        patience=3, mode='auto', verbose=False, \
        max_trigger=(100, 'epoch'))

    Trigger for Early Stopping in Multiple Node Environments

    It serves almost the same as
    :class:`~chainer.training.triggers.EarlyStoppingTrigger`,
    but it can correctly work in multiple node environments.

    The difference between it and
    :class:`~chainer.training.triggers.EarlyStoppingTrigger` is that,
    in each check interval, it computes the mean of the accumulated
    values *across all nodes*. In this way, all nodes will have the same
    value to determine the timing at which the trigger fires so that
    they will stop at the same time.

    Args:
        comm : ChainerMN communicator
        check_trigger: Trigger that decides the comparison
            interval between current best value and new value.
            This must be a tuple in the form of ``<int>,
            'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
        monitor (str) : The metric you want to monitor
        patience (int) : Counts to let the trigger be patient.
            The trigger will not fire until the condition is met
            for successive ``patience`` checks.
        mode (str) : ``'max'``, ``'min'``, or ``'auto'``.
            It is used to determine how to compare the monitored values.
        verbose (bool) : Enable verbose output.
            If verbose is true, you can get more information
        max_trigger: Upper bound of the number of training loops
        suffix (str): Suffix added to the name of the monitored
            metric after aggregation.

    .. note::
       ``patients`` is also available as an alias of ``patience`` for
       historical reason.
    """

    def __init__(self, comm,
                 *, check_trigger=(1, 'epoch'), monitor='main/loss',
                 patience=None, mode='auto', verbose=False,
                 max_trigger=(100, 'epoch'), suffix='_aggregated', **kwargs):

        # `patients` as an alias of `patience`
        monitor_aggregated = monitor + suffix

        self.actual_trigger = EarlyStoppingTrigger(check_trigger=check_trigger,
                                                   monitor=monitor_aggregated,
                                                   patience=patience,
                                                   mode=mode, verbose=verbose,
                                                   max_trigger=max_trigger,
                                                   **kwargs)
        self.aggregator = ObservationAggregator(
            comm, monitor,
            aggregated_key=monitor_aggregated,
            comm_trigger=check_trigger)

    def __call__(self, trainer):
        self.aggregator(trainer)
        return self.actual_trigger(trainer)

    def _stop_condition(self):
        return self.actual_trigger._stop_condition()

    def _init_summary(self):
        return self.actual_trigger._init_summary()

    def get_training_length(self):
        return self.actual_trigger.get_training_length()
