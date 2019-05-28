import numpy as np
from chainer.training import extension


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
         individual values. If not specified, `numpy.average` is used.
    """

    if aggregated_key is None:
        aggregated_key = original_key

    if aggregator is None:
        aggregator = np.average

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
