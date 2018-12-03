import chainer.cuda
import math
import mpi4py.MPI

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base
from chainermn import nccl


class TwoDimensionalCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm=mpi4py.MPI.COMM_WORLD):
        super(TwoDimensionalCommunicator, self).__init__(
            mpi_comm)
        if not nccl._available:
            raise RuntimeError(
                'NCCL is not available. '
                'Please confirm that NCCL is enabled in CuPy.'
            )

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.inter_mpi_comm = None
        self.intra_nccl_comm = None

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def _init_comms(self):
        if self.inter_mpi_comm is not None:
            assert self.intra_nccl_comm is not None
            return

        intra_mpi_comm = _communication_utility.init_intra_mpi_comm(
            self.mpi_comm, self.intra_rank, self.inter_rank)
        self.inter_mpi_comm = _communication_utility.init_inter_mpi_comm(
            self.mpi_comm, self.intra_rank, self.inter_rank)
        self.intra_nccl_comm = _communication_utility.init_nccl_comm(
            intra_mpi_comm)

    def allreduce_grad(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params_set_grad(model)
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_elems_per_node_2d = int(math.ceil(n_elems_total / self.size))
        n_elems_per_node_1d = n_elems_per_node_2d * self.inter_size
        n_bytes_per_node_1d = n_elems_per_node_1d * itemsize
        n_bytes_per_node_2d = n_elems_per_node_2d * itemsize
        n_bytes_buffer = n_bytes_per_node_2d * self.size

        self.gpu_buffer_a.assign(n_bytes_buffer)
        self.gpu_buffer_b.assign(n_bytes_buffer)
        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)

        # Intra-node reduce-scatter (1st dimension)
        self.intra_nccl_comm.reduceScatter(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(),
            n_elems_per_node_1d, nccl.NCCL_FLOAT, nccl.NCCL_SUM, stream.ptr)

        # Inter-node allreduce (2nd dimension)
        _communication_utility.inter_allreduce_gpu(
            self.inter_mpi_comm, self.size,
            self.gpu_buffer_a, self.gpu_buffer_b,
            n_bytes_per_node_1d, n_elems_per_node_2d,
            n_bytes_per_node_2d, stream)

        # Intra-node allgather (1st dimension)
        self.intra_nccl_comm.allGather(
            self.gpu_buffer_b.ptr(), self.gpu_buffer_a.ptr(),
            n_elems_per_node_1d, nccl.NCCL_FLOAT, stream.ptr)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)
