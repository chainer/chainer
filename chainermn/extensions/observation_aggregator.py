from __future__ import division
from chainer.training import extension, util


def observation_aggregator(comm, original_key, aggregated_key=None,
                           aggregator=None):
    """Returns an observation aggregator, which summarizes
    (computing the average, etc) the observation `original_key` in the trainer.
    It should be noted that, in each iteration, `original_key` must be reported
    either in every training process, or no process at all.
    Args:
         comm: ChainerMN communicator
         original_key (str): Key of the observation to be summarized.
         aggregated_key (str): Name of the key after the summarization.
         If not specified, it is set to `original_key` to overwrite it.
         aggregator (function): Function to compute summarization from
         individual values. If not specified, the average function is used.
    """

    if aggregated_key is None:
        aggregated_key = original_key

    if aggregator is None:
        def _average(xs):
            return sum(xs) / float(len(xs))
        aggregator = _average

    @extension.make_extension(
        trigger=(1, 'iteration'), priority=extension.PRIORITY_EDITOR)
    def _observation_aggregator(trainer):
        if original_key not in trainer.observation:
            return
        value = trainer.observation[original_key]
        value_collected = comm.gather_obj(value)

        if comm.rank == 0:
            assert len(value_collected) == comm.size
            value_summarized = aggregator(value_collected)
            comm.bcast_obj(value_summarized)
        else:
            value_summarized = comm.bcast_obj(None)

        trainer.observation[aggregated_key] = value_summarized

    return _observation_aggregator


class ObservationAggregator(extension.Extension):

    """Trainer extension to aggregate an observation in the trainer.
    Args:
         comm: ChainerMN communicator
         original_key (str): Key of the observation to be summarized.
         aggregated_key (str): Name of the key after the summarization.
         If not specified, it is set to `original_key` to overwrite it.
         comm_trigger: Trigger that decides the timing to communicate
         observation values for aggregation.
         aggregator (function): Function to compute summarization from
         individual values. It takes a list of lists of observed values.
         Each list contains all the observed values since the last communication.
    """

    trigger = 1, 'iteration'
    priority = extension.PRIORITY_EDITOR
    name = None

    def __init__(self, comm, original_key, aggregated_key=None,
                 comm_trigger=(1, 'iteration'), aggregator=None):
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
            self.observation_history.append(value)

        if not self.comm_trigger(trainer):
            return None

        internal_summary = self.internal_aggregator(self.observation_history)
        self.observation_history = []

        internal_summary_gathered = self.comm.gather_obj(internal_summary)

        if self.comm.rank == 0:
            global_summary = self.global_aggregator(internal_summary_gathered)
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
