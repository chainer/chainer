import math
import mpi4py.MPI
import numpy as np
import warnings

import chainer.cuda
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
        if nccl.get_version() < 2302:
            warnings.warn('NCCL 2.2 and older versions are deprecated.',
                          DeprecationWarning)

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

    def allreduce_grad(self, model, zero_fill=False):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params_set_grad(model, zero_fill)
        itemsize = 4
        n_elems_total = _memory_utility.count_grad_elements(params,
                                                            zero_fill)
        n_elems_per_node_2d = int(math.ceil(n_elems_total / self.size))
        n_elems_per_node_1d = n_elems_per_node_2d * self.inter_size
        n_bytes_per_node_1d = n_elems_per_node_1d * itemsize
        n_bytes_per_node_2d = n_elems_per_node_2d * itemsize
        n_bytes_buffer = n_bytes_per_node_2d * self.size

        self.gpu_buffer_a.assign(n_bytes_buffer)
        self.gpu_buffer_b.assign(n_bytes_buffer)

        allreduce_grad_dtype = np.float32

        _memory_utility.pack_params(
            params, 'grad', self.gpu_buffer_a, allreduce_grad_dtype, zero_fill)

        if chainer.is_debug():
            stream.synchronize()
            array_a = self.gpu_buffer_a.array(n_elems_total)
            array_b = self.gpu_buffer_b.array(n_elems_total)
            self.check_ready_to_allreduce(array_a, array_b)

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

        if chainer.is_debug():
            stream.synchronize()
            self.ensure_all_finite(self.gpu_buffer_a.array(n_elems_total))

        _memory_utility.unpack_params(
            params, 'grad', self.gpu_buffer_a, allreduce_grad_dtype, zero_fill)
