import mpi4py.MPI
import numpy as np

import chainer
from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class FlatCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(FlatCommunicator, self).__init__(mpi_comm)

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def allreduce_grad(self, model):
        params = _memory_utility.extract_params_set_grad(model)
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)
        self.gpu_buffer_b.assign(n_bytes_total)

        allreduce_grad_dtype = np.float32

        _memory_utility.pack_params(
            params, 'grad', self.gpu_buffer_a, allreduce_grad_dtype)

        self.mpi_comm.Allreduce(
            [self.gpu_buffer_a.buffer(n_bytes_total), mpi4py.MPI.FLOAT],
            [self.gpu_buffer_b.buffer(n_bytes_total), mpi4py.MPI.FLOAT])
        arr = self.gpu_buffer_b.array(n_elems_total)
        arr *= (1.0 / self.size)

        _memory_utility.unpack_params(
            params, 'grad', self.gpu_buffer_b, allreduce_grad_dtype)
