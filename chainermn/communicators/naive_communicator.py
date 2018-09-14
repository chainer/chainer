import mpi4py.MPI

from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class NaiveCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def allreduce_grad(self, model):
        for param in _memory_utility.extract_params_set_grad(model):
            buf = _memory_utility.array_to_buffer_object(param.grad)
            self.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, buf)
            param.grad /= self.size
