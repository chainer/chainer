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
        if not nccl._available or nccl.get_version() < 2000:
            raise RuntimeError(
                'PureNcclCommunicator is only supported on NCCL 2.0+')

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
        self.div_by_size = None
        self.params_data = None

    def _init_comms(self):
        if self.nccl_comm is not None:
            return
        self.nccl_comm = _communication_utility.init_nccl_comm(self.mpi_comm)

    def bcast_data(self, model):
        self._init_comms()
        params = _memory_utility.extract_params_set_data(model)
        data_dtype = _get_param_data_dtype(params[0])
        n_elems = sum(param.data.size for param in params)
        data_grad_n_bytes = data_dtype.itemsize * n_elems
        if self.gpu_tmp_buffer.size != data_grad_n_bytes:
            self.gpu_tmp_buffer.assign(data_grad_n_bytes)
        stream = chainer.cuda.Stream.null

        _memory_utility.pack_params(
            params, data_dtype.itemsize, 'data',
            self.gpu_tmp_buffer, stream)
        self.nccl_comm.bcast(self.gpu_tmp_buffer.ptr(), n_elems,
                             _get_nccl_type_id(data_dtype), 0, stream.ptr)
        _memory_utility.unpack_params(
            params, data_dtype.itemsize, 'data',
            self.gpu_tmp_buffer, stream)

    def allreduce_grad(self, model):
        stream = chainer.cuda.Stream.null
        self._allreduce_grad_async(model, stream)

    def _allreduce_grad_async(self, model, stream):
        self._init_comms()
        params = _memory_utility.extract_params_set_grad(model)
        grad_dtype = _get_param_grad_dtype(params[0])
        if self.allreduce_grad_dtype is None:
            allreduce_grad_dtype = grad_dtype
        else:
            allreduce_grad_dtype = self.allreduce_grad_dtype
        n_elems = sum(param.grad.size for param in params)
        needs_sync = self._assign_for_allreduce_grad(grad_dtype,
                                                     allreduce_grad_dtype,
                                                     n_elems)
        if stream != chainer.cuda.Stream.null and needs_sync:
            chainer.cuda.Stream.null.synchronize()

        self._pack_params_to_buffer(params, grad_dtype, allreduce_grad_dtype,
                                    n_elems, stream)
        self.nccl_comm.allReduce(self.gpu_buffer_a.ptr(),
                                 self.gpu_buffer_b.ptr(), n_elems,
                                 _get_nccl_type_id(allreduce_grad_dtype),
                                 nccl.NCCL_SUM,
                                 stream.ptr)
        if self.div_by_size is None:
            self.div_by_size = chainer.cuda.cupy.ElementwiseKernel(
                '{} x'.format(allreduce_grad_dtype.name),
                '{} y'.format(allreduce_grad_dtype.name),
                'y = x*(1.0/{})'.format(self.size), 'div_by_size')
        self.div_by_size(
            self.gpu_buffer_b.array(n_elems,
                                    dtype=allreduce_grad_dtype),
            self.gpu_buffer_a.array(n_elems,
                                    dtype=allreduce_grad_dtype),
            stream=stream)
        self._unpack_params_from_buffer(params, grad_dtype,
                                        allreduce_grad_dtype, n_elems, stream)

    def _assign_for_allreduce_grad(self, grad_dtype, allreduce_grad_dtype,
                                   n_elems):
        allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
        needs_sync = False
        if self.gpu_buffer_a.size != allreduce_grad_n_bytes:
            self.gpu_buffer_a.assign(allreduce_grad_n_bytes)
            needs_sync = True
        if self.gpu_buffer_b.size != allreduce_grad_n_bytes:
            self.gpu_buffer_b.assign(allreduce_grad_n_bytes)
            needs_sync = True

        if grad_dtype != allreduce_grad_dtype:
            grad_n_bytes = grad_dtype.itemsize * n_elems
            if self.gpu_tmp_buffer.size != grad_n_bytes:
                self.gpu_tmp_buffer.assign(grad_n_bytes)
                needs_sync = True
        return needs_sync

    def _pack_params_to_buffer(self, params, grad_dtype, allreduce_grad_dtype,
                               n_elems, stream):
        if self.batched_copy:
            params_data = _ParamsData(params, 'grad')
            _batched_pack_params(params_data, self.gpu_buffer_a,
                                 allreduce_grad_dtype)
            self.params_data = params_data
            # self.params_data will be re-used by _unpack_params_from_buffer
            return
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_buffer_a, stream=stream)
        else:
            if self.grad_dtype_to_allreduce_dtype_kernel is None:
                self.grad_dtype_to_allreduce_dtype_kernel = \
                    _get_converting_kernel(
                        grad_dtype, allreduce_grad_dtype,
                        'grad_dtype_to_allreduce_dtype_kernel')

            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_tmp_buffer, stream=stream)

            self.grad_dtype_to_allreduce_dtype_kernel(
                self.gpu_tmp_buffer.array(n_elems, dtype=grad_dtype),
                self.gpu_buffer_a.array(n_elems,
                                        dtype=allreduce_grad_dtype),
                stream=stream)

    def _unpack_params_from_buffer(self, params, grad_dtype,
                                   allreduce_grad_dtype, n_elems, stream):
        if self.batched_copy:
            if self.params_data is not None:
                params_data = self.params_data
                self.params_data = None
            else:
                params_data = _ParamsData(params, 'grad')
            _batched_unpack_params(params_data, self.gpu_buffer_a,
                                   allreduce_grad_dtype)
            return
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.unpack_params(
                params, allreduce_grad_dtype.itemsize, 'grad',
                self.gpu_buffer_a, stream)

        else:
            if self.allreduce_dtype_to_grad_dtype_kernel is None:
                self.allreduce_dtype_to_grad_dtype_kernel = \
                    _get_converting_kernel(
                        allreduce_grad_dtype, grad_dtype,
                        'allreduce_dtype_to_grad_dtype_kernel')
            self.allreduce_dtype_to_grad_dtype_kernel(
                self.gpu_buffer_a.array(n_elems,
                                        dtype=allreduce_grad_dtype),
                self.gpu_tmp_buffer.array(n_elems, dtype=grad_dtype),
                stream=stream)

            _memory_utility.unpack_params(
                params, grad_dtype.itemsize, 'grad', self.gpu_tmp_buffer,
                stream=stream)


def _get_converting_kernel(src_dtype, dst_dtype, kernel_name):
    return chainer.cuda.cupy.ElementwiseKernel(
        '{} x'.format(src_dtype.name),
        '{} y'.format(dst_dtype.name),
        'y = x', kernel_name)


def _get_param_data_dtype(param):
    return param.data.dtype


def _get_param_grad_dtype(param):
    return param.grad.dtype


def _get_nccl_type_id(dtype):
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCCL_FLOAT64
    else:
        raise ValueError(
            'dtype must be float16, float32, or float64.')


class _ParamsData(object):
    def __init__(self, params, attr_name):
        n_params = len(params)
        params_dptr = np.empty(n_params, dtype=np.int64)
        params_dtype = np.empty(n_params, dtype=np.int32)
        params_size_csum = np.empty(n_params+1, dtype=np.int32)
        params_size_csum[0] = 0
        for i, param in enumerate(params):
            v = getattr(param, attr_name)
            params_dptr[i] = v.data.ptr
            if v.dtype not in [np.float16, np.float32]:
                raise ValueError('dtype must be float16 or float32.')
            params_dtype[i] = _get_nccl_type_id(v.dtype)
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
    buf_dtype = _get_nccl_type_id(dtype)
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
    buf_dtype = _get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    _cupy_batched_unpack_params()(
        (n_blocks, ), (n_threads, ),
        (buffer.memory.ptr, buf_dtype, n_elems,
         params_dptr, params_dtype, params_size_csum, n_params))


def _cupy_batched_pack_params():
    return chainer.cuda.cupy.RawKernel(r'''
#include <cuda_fp16.h>
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
#include <cuda_fp16.h>
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
