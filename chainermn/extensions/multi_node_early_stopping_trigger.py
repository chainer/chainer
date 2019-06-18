import random, string
from chainer.training.triggers import EarlyStoppingTrigger
from chainermn.extensions import ObservationAggregator


def _random_suffix(n):
    return ''.join(random.choices(string.ascii_letters, k=n))


class MultiNodeEarlyStoppingTrigger(object):

    def __init__(self, comm, check_trigger=(1, 'epoch'), monitor='main/loss',
                 patience=None, mode='auto', verbose=False,
                 max_trigger=(100, 'epoch'), **kwargs):

        # `patients` as an alias of `patience`
        monitor_aggregated = monitor + '-aggregated-' + _random_suffix(10)

        self.actual_trigger = EarlyStoppingTrigger(check_trigger=check_trigger,
                                              monitor=monitor_aggregated,
                                              patience=patience,
                                              mode=mode, verbose=verbose,
                                              max_trigger=max_trigger,
                                              **kwargs)
        self.aggregator = ObservationAggregator(comm, monitor, monitor_aggregated,
                                           check_trigger)

    def __call__(self, trainer):
        self.aggregator(trainer)
        return self.actual_trigger(trainer)

    def _stop_condition(self):
        return self.actual_trigger._stop_condition()

    def _init_summary(self):
        return self.actual_trigger._init_summary()

    def get_training_length(self):
        return self.actual_trigger.get_training_length()
