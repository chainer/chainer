import numpy as np

from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class FlatCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(FlatCommunicator, self).__init__(mpi_comm)

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def multi_node_mean_grad(self, model, zero_fill=False):
        params = _memory_utility.extract_params_set_grad(model, zero_fill)
        itemsize = 4
        n_elems_total = _memory_utility.count_grad_elements(params,
                                                            zero_fill)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)
        self.gpu_buffer_b.assign(n_bytes_total)

        allreduce_grad_dtype = np.float32

        self._pack_params_to_buffer(params, 'grad', buffer=self.gpu_buffer_a,
                                    allreduce_grad_dtype=allreduce_grad_dtype,
                                    zero_fill=zero_fill)

        self._multi_node_mean(self.gpu_buffer_a.array(n_elems_total),
                              self.gpu_buffer_b.array(n_elems_total))

        self._unpack_params_from_buffer(params, 'grad', self.gpu_buffer_b,
                                        allreduce_grad_dtype, zero_fill)
