from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base
import numpy as np


class DummyCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    """Dummy communicator that does not communicate at all.

    This class is intended to measure the overhead of packing and unpacking.
    This class does not pass the tests.
    """

    def __init__(self, mpi_comm):
        super(DummyCommunicator, self).__init__(mpi_comm)

        self.gpu_buffer_a = _memory_utility.DeviceMemory()

    def multi_node_mean_grad(self, model, zero_fill=False):
        params = _memory_utility.extract_params_set_grad(model, zero_fill)
        itemsize = 4
        n_elems_total = _memory_utility.count_grad_elements(params,
                                                            zero_fill)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)

        self._pack_params_to_buffer(params, 'grad',
                                    buffer=self.gpu_buffer_a,
                                    allreduce_grad_dtype=np.float32,
                                    zero_fill=zero_fill)

        self._unpack_params_from_buffer(params, 'grad',
                                        buffer=self.gpu_buffer_a,
                                        allreduce_grad_dtype=np.float32,
                                        zero_fill=zero_fill)
