from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class NaiveCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def multi_node_mean_grad(self, model, zero_fill=False):
        params = _memory_utility.extract_params_set_grad(model, zero_fill)
        for param in params:
            if zero_fill and param.grad is None:
                if param.data is None:
                    continue
                param.grad = param.xp.zeros_like(param.data)
            self._multi_node_mean(None, param.grad)
