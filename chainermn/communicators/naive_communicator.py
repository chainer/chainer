import mpi4py.MPI
import numpy as np

from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class NaiveCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def allreduce_grad(self, model):
        for param in _memory_utility.extract_params_set_grad(model):
            grad = param.grad
            is_float16 = param.grad.dtype == np.float16
            if is_float16:
                grad = grad.astype(np.float32)
            buf = _memory_utility.array_to_buffer_object(grad)
            self.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, buf)
            if is_float16:
                param.grad = grad.astype(np.float16)
            param.grad /= self.size
