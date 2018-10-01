import chainer.cuda

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base
from chainermn import nccl


class SingleNodeCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(SingleNodeCommunicator, self).__init__(mpi_comm)

        if self.inter_size != 1:
            raise ValueError('SingleNodeCommunicator cannot be used under '
                             'multi-node settings')
        if not nccl._available:
            raise RuntimeError(
                'NCCL is not available. '
                'Please confirm that NCCL is enabled in CuPy.'
            )
        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.intra_nccl_comm = None

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def _init_comms(self):
        if self.intra_nccl_comm is not None:
            return

        intra_mpi_comm = _communication_utility.init_intra_mpi_comm(
            self.mpi_comm, self.intra_rank, self.inter_rank)
        self.intra_nccl_comm = _communication_utility.init_nccl_comm(
            intra_mpi_comm)

    def bcast_data(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)

        _memory_utility.pack_params(
            params, itemsize, 'data', self.gpu_buffer_a)

        self.intra_nccl_comm.bcast(
            self.gpu_buffer_a.ptr(), n_elems_total, nccl.NCCL_FLOAT,
            0, stream.ptr)

        _memory_utility.unpack_params(
            params, itemsize, 'data', self.gpu_buffer_a)

    def allreduce_grad(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params_set_grad(model)
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)
        self.gpu_buffer_b.assign(n_bytes_total)

        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)

        self.intra_nccl_comm.allReduce(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(), n_elems_total,
            nccl.NCCL_FLOAT, nccl.NCCL_SUM, stream.ptr)

        arr = self.gpu_buffer_b.array(n_elems_total)
        arr *= (1.0 / self.size)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
