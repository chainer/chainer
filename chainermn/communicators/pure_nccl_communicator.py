import warnings

import chainer.cuda

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base
from chainermn import nccl

import numpy as np


class PureNcclCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm, allreduce_grad_dtype=None,
                 batched_copy=False):
        super(PureNcclCommunicator, self).__init__(mpi_comm)
        if not nccl._available or nccl.get_build_version() < 2000:
            raise RuntimeError(
                'PureNcclCommunicator is only supported on NCCL 2.0+')

        if nccl.get_version() < 2302:
            warnings.warn('NCCL 2.2 and older versions are deprecated.',
                          DeprecationWarning)

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.nccl_comm = None

        self.gpu_tmp_buffer = _memory_utility.DeviceMemory()
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

        if allreduce_grad_dtype is not None:
            self.allreduce_grad_dtype = np.dtype(allreduce_grad_dtype)
            if self.allreduce_grad_dtype.kind != 'f':
                raise ValueError(
                    'allreduce_grad_dtype must be'
                    'numpy.float16, numpy.float32,'
                    'numpy.float64, or None.')
        else:
            self.allreduce_grad_dtype = None
        self.batched_copy = batched_copy
        self.grad_dtype_to_allreduce_dtype_kernel = None
        self.allreduce_dtype_to_grad_dtype_kernel = None
        if batched_copy:
            self.params_data = _memory_utility.ExtractedParamAttrsGpu()
        else:
            self.params_data = None

    def _init_comms(self):
        if self.nccl_comm is not None:
            return
        self.nccl_comm = _communication_utility.init_nccl_comm(self.mpi_comm)

    def bcast_data(self, model):
        self._init_comms()
        params = _memory_utility.extract_params_set_data(model)
        data_dtype = chainer.get_dtype()
        n_elems = sum(param.data.size for param in params)
        data_grad_n_bytes = data_dtype.itemsize * n_elems
        if self.gpu_tmp_buffer.size != data_grad_n_bytes:
            self.gpu_tmp_buffer.assign(data_grad_n_bytes)
        stream = chainer.cuda.Stream.null

        _memory_utility.pack_params(
            params, 'data', self.gpu_tmp_buffer, data_dtype, False, stream)
        self.nccl_comm.bcast(self.gpu_tmp_buffer.ptr(), n_elems,
                             _communication_utility._get_nccl_type_id(
                                 data_dtype),
                             0, stream.ptr)
        _memory_utility.unpack_params(
            params, 'data', self.gpu_tmp_buffer, data_dtype, False, stream)

    def allreduce_grad(self, model, zero_fill=False):
        stream = chainer.cuda.Stream.null
        self._allreduce_grad_async(model, zero_fill, stream)

    def _allreduce_grad_async(self, model, zero_fill, stream):
        self._init_comms()
        params = _memory_utility.extract_params_set_grad(model, zero_fill)

        # NOTE: we need to explicitly check `is None` , becuase
        # numpy's dtype object is evaluated to False in numpy <= 1.12.1
        if self.allreduce_grad_dtype is not None:
            allreduce_grad_dtype = self.allreduce_grad_dtype
        else:
            allreduce_grad_dtype = chainer.get_dtype()

        assert allreduce_grad_dtype is not None

        n_elems = _memory_utility.count_grad_elements(params,
                                                      zero_fill)
        needs_sync = _memory_utility.prepare_multi_node_mean_pack_buffer(
            allreduce_grad_dtype, n_elems, self.gpu_buffer_a,
            self.gpu_buffer_b)
        if stream != chainer.cuda.Stream.null and needs_sync:
            chainer.cuda.Stream.null.synchronize()

        # pack grads from params -> buffer A
        _memory_utility.pack_params_to_buffer(params, self.params_data,
                                              self.gpu_buffer_a,
                                              allreduce_grad_dtype,
                                              zero_fill, stream)

        # Mean from buffer A -> buffer B
        self.multi_node_mean_nccl(self.gpu_buffer_a, self.gpu_buffer_b,
                                  n_elems,
                                  allreduce_grad_dtype, stream)

        # unpack params from buffer B -> params
        _memory_utility.unpack_params_from_buffer(params, self.params_data,
                                                  self.gpu_buffer_b,
                                                  allreduce_grad_dtype,
                                                  zero_fill, stream)

    def multi_node_mean_nccl(self, gpu_buffer_a, gpu_buffer_b,
                             n_elems, dtype, stream=None):
        # Performs allreduce and division by size, i.e. mean.
        # # Sigma(gpu_buffer_a, all-procs)/n -> gpu_buffer_b
        if chainer.is_debug():
            stream.synchronize()
            array_a = gpu_buffer_a.array(n_elems, dtype=dtype)
            array_b = gpu_buffer_b.array(n_elems, dtype=dtype)
            self.check_ready_to_allreduce(array_a, array_b)

        if stream is None:
            stream = chainer.cuda.Stream.null
        self._init_comms()
        type_id = _communication_utility._get_nccl_type_id(dtype)
        self.nccl_comm.allReduce(gpu_buffer_a.ptr(),
                                 gpu_buffer_b.ptr(), n_elems,
                                 type_id, nccl.NCCL_SUM, stream.ptr)
        div_by_size = chainer.cuda.cupy.ElementwiseKernel(
            '',
            '{} x'.format(dtype.name),
            'x *= (1.0/{})'.format(self.size), 'div_by_size')
        div_by_size(
            gpu_buffer_b.array(n_elems, dtype=dtype),
            stream=stream)

        if chainer.is_debug():
            stream.synchronize()
            self.ensure_all_finite(gpu_buffer_a.array(n_elems, dtype=dtype))
