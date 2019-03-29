from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base


class DummyCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    """Dummy communicator that does not communicate at all.

    This class is intended to measure the overhead of packing and unpacking.
    This class does not pass the tests.
    """

    def __init__(self, mpi_comm):
        super(DummyCommunicator, self).__init__(mpi_comm)

        self.gpu_buffer_a = _memory_utility.DeviceMemory()

    def allreduce_grad(self, model):
        params = _memory_utility.extract_params_set_grad(model)
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)

        _memory_utility.pack_params(
            params, 'grad', self.gpu_buffer_a)

        _memory_utility.unpack_params(
            params, 'grad', self.gpu_buffer_a)
