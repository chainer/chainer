from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class NaiveCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def allreduce_grad(self, model):
        for param in _memory_utility.extract_params_set_grad(model):
            self.multi_node_mean(None, param.grad)
