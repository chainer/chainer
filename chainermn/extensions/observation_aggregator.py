from __future__ import division
from chainer.training import extension, util
from chainer import Variable


class ObservationAggregator(extension.Extension):

    """Trainer extension to aggregate an observation in the trainer.
    Args:
         comm: ChainerMN communicator
         original_key (str): Key of the observation to be summarized.
         If the observation is a :class:`chainer.Variable`, its value
         is automatically copied to CPU.
         aggregated_key (str): Name of the key after the summarization.
         If not specified, it is set to `original_key` to overwrite it.
         comm_trigger: Trigger that decides the timing to communicate
         observation values for aggregation.
         aggregator (function): Function to compute summarization from
         individual values. It takes a list of lists of observed values.
         Each list contains all the observed values since
         the last communication.
    """

    trigger = 1, 'iteration'
    priority = extension.PRIORITY_EDITOR
    name = None

    def __init__(self, comm, original_key, aggregated_key=None,
                 *, comm_trigger=(1, 'iteration'), aggregator=None):
        self.comm = comm
        self.original_key = original_key

        if aggregated_key is None:
            self.aggregated_key = original_key
        else:
            self.aggregated_key = aggregated_key

        self.comm_trigger = util.get_trigger(comm_trigger)
        self.observation_history = []

        self.aggregator = aggregator or _average_2d

    def compute_summary(self, trainer):
        if self.original_key in trainer.observation:
            value = trainer.observation[self.original_key]
            if isinstance(value, Variable):
                # use to native device as ChainerX array cannot
                # be converted to numpy directly, which is what `to_cpu()` does
                value.to_device("native")
            self.observation_history.append(value)

        if not self.comm_trigger(trainer):
            return None

        observation_history_gathered = self.comm.gather_obj(
            self.observation_history)
        self.observation_history = []

        if self.comm.rank == 0:
            global_summary = self.aggregator(observation_history_gathered)
            self.comm.bcast_obj(global_summary)
        else:
            global_summary = self.comm.bcast_obj(None)

        return global_summary

    def __call__(self, trainer):
        summary = self.compute_summary(trainer)

        if summary is not None:
            trainer.observation[self.aggregated_key] = summary


def _average_2d(xs):
    xs = sum(xs, [])
    return sum(xs) / len(xs)
