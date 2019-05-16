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
        needs_sync = self._prepare_allreduce_pack_buffer(allreduce_grad_dtype,
                                                         n_elems)
        if stream != chainer.cuda.Stream.null and needs_sync:
            chainer.cuda.Stream.null.synchronize()

        # pack grads from params -> buffer A
        self._pack_params_to_buffer(params, allreduce_grad_dtype,
                                    zero_fill, stream)

        # Allreduce from buffer A -> buffer B
        # div by comm_size from buffer B -> buffer A
        self.multi_node_mean_nccl(self.gpu_buffer_a, self.gpu_buffer_b,
                                  n_elems,
                                  allreduce_grad_dtype, stream)

        # unpack params from buffer A -> params
        self._unpack_params_from_buffer(params, allreduce_grad_dtype,
                                        zero_fill, stream)

    def _prepare_allreduce_pack_buffer(self, allreduce_grad_dtype, n_elems):
        allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
        needs_sync = False

        if self.gpu_buffer_a.size != allreduce_grad_n_bytes:
            self.gpu_buffer_a.assign(allreduce_grad_n_bytes)
            needs_sync = True
        if self.gpu_buffer_b.size != allreduce_grad_n_bytes:
            self.gpu_buffer_b.assign(allreduce_grad_n_bytes)
            needs_sync = True

        return needs_sync

    def _pack_params_to_buffer(self, params, allreduce_grad_dtype,
                               zero_fill, stream):
        if self.batched_copy:
            params_data = _ParamsData(params, 'grad', zero_fill)
            _batched_pack_params(params_data, self.gpu_buffer_a,
                                 allreduce_grad_dtype)
            self.params_data = params_data
            # self.params_data will be re-used by _unpack_params_from_buffer
        else:
            _memory_utility.pack_params(
                params, 'grad',
                self.gpu_buffer_a,
                transfer_dtype=allreduce_grad_dtype,
                zero_fill=zero_fill,
                stream=stream)

    def _unpack_params_from_buffer(self, params,
                                   allreduce_grad_dtype, zero_fill, stream):
        if self.batched_copy:
            if self.params_data is not None:
                params_data = self.params_data
                self.params_data = None
            else:
                params_data = _ParamsData(params, 'grad', zero_fill)
            _batched_unpack_params(params_data, self.gpu_buffer_a,
                                   allreduce_grad_dtype)
            return
        else:
            _memory_utility.unpack_params(
                params, 'grad', self.gpu_buffer_a,
                allreduce_grad_dtype, zero_fill, stream)

    def multi_node_mean_nccl(self, gpu_buffer_a, gpu_buffer_b,
                             n_elems, dtype, stream=None):
        # Performs allreduce and division by size, i.e. mean.
        # gpu_buffer_a = Sigma(gpu_buffer_a, all-procs) / self.size
        # b is just used as buffer
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
            '{} x'.format(dtype.name),
            '{} y'.format(dtype.name),
            'y = x*(1.0/{})'.format(self.size), 'div_by_size')
        div_by_size(
            gpu_buffer_b.array(n_elems, dtype=dtype),
            gpu_buffer_a.array(n_elems, dtype=dtype),
            stream=stream)

        if chainer.is_debug():
            stream.synchronize()
            self.ensure_all_finite(gpu_buffer_a.array(n_elems, dtype=dtype))


def _get_converting_kernel(src_dtype, dst_dtype, kernel_name):
    return chainer.cuda.cupy.ElementwiseKernel(
        '{} x'.format(src_dtype.name),
        '{} y'.format(dst_dtype.name),
        'y = x', kernel_name)


def _get_param_data_dtype(param):
    return param.data.dtype


def _get_param_grad_dtype(param):
    return param.grad.dtype


class _ParamsData(object):
    def __init__(self, params, attr_name, zero_fill):
        n_params = len(params)
        params_dptr = np.empty(n_params, dtype=np.int64)
        params_dtype = np.empty(n_params, dtype=np.int32)
        params_size_csum = np.empty(n_params+1, dtype=np.int32)
        params_size_csum[0] = 0
        for i, param in enumerate(params):
            v = getattr(param, attr_name)
            if attr_name == 'grad' and v is None and zero_fill:
                v = param.xp.zeros_like(param.data)
                setattr(param, attr_name, v)
            params_dptr[i] = v.data.ptr
            if v.dtype not in [np.float16, np.float32]:
                raise ValueError('dtype must be float16 or float32.')
            params_dtype[i] = _communication_utility._get_nccl_type_id(v.dtype)
            params_size_csum[i+1] = params_size_csum[i] + v.size
        self.n_params = n_params
        self.n_elems = params_size_csum[n_params]
        self.size_csum = chainer.cuda.cupy.asarray(params_size_csum)
        self.dtype = chainer.cuda.cupy.asarray(params_dtype)
        self.dptr = chainer.cuda.cupy.asarray(params_dptr)


def _batched_pack_params(params_data, buffer, dtype):
    n_params = params_data.n_params
    n_elems = params_data.n_elems
    params_dptr = params_data.dptr
    params_dtype = params_data.dtype
    params_size_csum = params_data.size_csum
    buf_dtype = _communication_utility._get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    _cupy_batched_pack_params()(
        (n_blocks, ), (n_threads, ),
        (buffer.memory.ptr, buf_dtype, n_elems,
         params_dptr, params_dtype, params_size_csum, n_params))


def _batched_unpack_params(params_data, buffer, dtype):
    n_params = params_data.n_params
    n_elems = params_data.n_elems
    params_dptr = params_data.dptr
    params_dtype = params_data.dtype
    params_size_csum = params_data.size_csum
    buf_dtype = _communication_utility._get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    _cupy_batched_unpack_params()(
        (n_blocks, ), (n_threads, ),
        (buffer.memory.ptr, buf_dtype, n_elems,
         params_dptr, params_dtype, params_size_csum, n_params))


def _cupy_batched_pack_params():
    return chainer.cuda.cupy.RawKernel(r'''
#include <cupy/carray.cuh>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
    extern "C" __global__
    void cupy_batched_pack_params(
            void *dst0, int dst_dtype, int n_elems,
            unsigned long *params_dptr, int *params_dtype,
            int *params_size_csum, int n_params) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n_elems) return;
        int j_min = 0;
        int j_max = n_params - 1;
        int j;
        while (1) {
            j = (j_min + j_max) / 2;
            if (tid < params_size_csum[j]) {
                j_max = j - 1;
                continue;
            }
            if (tid >= params_size_csum[j+1]){
                j_min = j + 1;
                continue;
            }
            break;
        }
        assert(tid >= params_size_csum[j]);
        assert(tid < params_size_csum[j+1]);
        int src_dtype = params_dtype[j];
        int src_idx = tid - params_size_csum[j];
        if (dst_dtype == NCCL_FLOAT16) {
            half* dst = (half*) dst0;
            if (src_dtype == NCCL_FLOAT16) {
                dst[tid] = (half) (((half*) (params_dptr[j]))[src_idx]);
            }
            else if (src_dtype == NCCL_FLOAT32) {
                dst[tid] = (half) (((float*) (params_dptr[j]))[src_idx]);
            }
        }
        else if (dst_dtype == NCCL_FLOAT32) {
            float* dst = (float*) dst0;
            if (src_dtype == NCCL_FLOAT16) {
                dst[tid] = (float) (((half*) (params_dptr[j]))[src_idx]);
            }
            else if (src_dtype == NCCL_FLOAT32) {
                dst[tid] = (float) (((float*) (params_dptr[j]))[src_idx]);
            }
       }
    }
    ''', 'cupy_batched_pack_params')


def _cupy_batched_unpack_params():
    return chainer.cuda.cupy.RawKernel(r'''
#include <cupy/carray.cuh>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
    extern "C" __global__
    void cupy_batched_unpack_params(
            void *src0, int src_dtype, int n_elems,
            unsigned long *params_dptr, int *params_dtype,
            int *params_size_csum, int n_params) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n_elems) return;
        int j_min = 0;
        int j_max = n_params - 1;
        int j;
        while (1) {
            j = (j_min + j_max) / 2;
            if (tid < params_size_csum[j]) {
                j_max = j - 1;
                continue;
            }
            if (tid >= params_size_csum[j+1]){
                j_min = j + 1;
                continue;
            }
            break;
        }
        assert(tid >= params_size_csum[j]);
        assert(tid < params_size_csum[j+1]);
        int dst_dtype = params_dtype[j];
        int dst_idx = tid - params_size_csum[j];
        if (src_dtype == NCCL_FLOAT16) {
            half* src = (half*) src0;
            if (dst_dtype == NCCL_FLOAT16) {
                ((half*) (params_dptr[j]))[dst_idx] = (half) src[tid];
            }
            else if (dst_dtype == NCCL_FLOAT32) {
                ((float*) (params_dptr[j]))[dst_idx] = (float) src[tid];
            }
        }
        else if (src_dtype == NCCL_FLOAT32) {
            float* src = (float*) src0;
            if (dst_dtype == NCCL_FLOAT16) {
                ((half*) (params_dptr[j]))[dst_idx] = (half) src[tid];
            }
            else if (dst_dtype == NCCL_FLOAT32) {
                ((float*) (params_dptr[j]))[dst_idx] = (float) src[tid];
            }
       }
    }''', 'cupy_batched_unpack_params')
