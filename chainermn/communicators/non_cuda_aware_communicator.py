import warnings

import chainer.cuda
import math
import mpi4py.MPI
import numpy as np

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base
from chainermn import nccl


class NonCudaAwareCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NonCudaAwareCommunicator, self).__init__(mpi_comm)
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
        self.cpu_buffer_a = _memory_utility.HostPinnedMemory()
        self.cpu_buffer_b = _memory_utility.HostPinnedMemory()

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

    def bcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            if param.data is not None:
                data = param.data
                tmp_cpu = chainer.cuda.to_cpu(data)

                is_float16 = tmp_cpu.dtype == np.float16
                if is_float16:
                    tmp_cpu = tmp_cpu.astype(np.float32)

                self.mpi_comm.Bcast(tmp_cpu)
                if is_float16:
                    tmp_cpu = tmp_cpu.astype(np.float16)

                tmp_gpu = chainer.cuda.to_gpu(tmp_cpu)
                data[:] = tmp_gpu

    def allreduce_grad(self, model, zero_fill=False):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params_set_grad(model, zero_fill)
        itemsize = 4
        n_elems_total = _memory_utility.count_grad_elements(params,
                                                            zero_fill)
        n_elems_per_node = int(math.ceil(n_elems_total / self.inter_size))
        n_elems_buffer = n_elems_per_node * self.inter_size
        n_bytes_per_node = n_elems_per_node * itemsize
        n_bytes_buffer = n_bytes_per_node * self.inter_size

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

        # Intra-node reduce
        self.intra_nccl_comm.reduce(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(), n_elems_total,
            nccl.NCCL_FLOAT, nccl.NCCL_SUM, 0, stream.ptr)

        # Inter-node allreduce
        if self.intra_rank == 0:
            self.cpu_buffer_a.assign(n_bytes_buffer)
            self.cpu_buffer_b.assign(n_bytes_buffer)

            arr_b = self.gpu_buffer_b.array(n_elems_buffer)
            arr_b.data.copy_to_host(self.cpu_buffer_b.ptr(), n_bytes_buffer)

            self.inter_mpi_comm.Alltoall(
                [self.cpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT],
                [self.cpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])

            # Reduction in GPU
            arr_a = self.gpu_buffer_a.array(n_elems_buffer)
            arr_a.data.copy_from_host(self.cpu_buffer_a.ptr(), n_bytes_buffer)
            arr_a = arr_a.reshape(self.inter_size, n_elems_per_node)
            arr_a = arr_a.sum(axis=0)
            arr_a *= 1.0 / self.size
            arr_a.data.copy_to_host(self.cpu_buffer_a.ptr(), n_bytes_per_node)

            self.inter_mpi_comm.Allgather(
                [self.cpu_buffer_a.buffer(n_bytes_per_node), mpi4py.MPI.FLOAT],
                [self.cpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])

            arr_b.data.copy_from_host(self.cpu_buffer_b.ptr(), n_bytes_buffer)

        # Intra-node bcast
        self.intra_nccl_comm.bcast(
            self.gpu_buffer_b.ptr(), n_elems_total, nccl.NCCL_FLOAT, 0,
            stream.ptr)

        if chainer.is_debug():
            stream.synchronize()
            self.ensure_all_finite(self.gpu_buffer_b.array(n_elems_total))

        _memory_utility.unpack_params(
            params, 'grad', self.gpu_buffer_b, allreduce_grad_dtype, zero_fill)
