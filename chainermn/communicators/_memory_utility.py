import ctypes

import mpi4py.MPI
import numpy as np

from chainermn import nccl

import chainer.backends
try:
    import cupy as cp
    _cupy_avail = True
except ImportError:
    _cupy_avail = False

from cupy.cuda import memory


class HostPinnedMemory(object):

    def __init__(self):
        if not _cupy_avail:
            raise RuntimeError("HostPinnedMemory cannot be used: " +
                               "Cupy is not available.")
        self.size = 0
        self.memory = None

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc_pinned_memory(size)

    def ptr(self, offset=0):
        return ctypes.c_void_p(self.memory.ptr + offset)

    def buffer(self, size):
        return ctypes.cast(
            self.memory.ptr,
            ctypes.POINTER(ctypes.c_ubyte * size)
        ).contents

    def array(self, count, offset=0, dtype=np.float32):
        if dtype is None:
            raise TypeError('dtype must be an instance of numpy.dtype class')
        return np.frombuffer(
            self.memory, count=count, offset=offset, dtype=dtype)


class DeviceMemory(object):

    def __init__(self):
        if not _cupy_avail:
            raise RuntimeError("DeviceMemory cannot be used: " +
                               "Cupy is not available.")
        self.size = 0
        self.memory = None

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc(size)

    def from_device(self, src, size, offset=0, stream=None):
        dst = self.memory + offset
        if stream is None:
            dst.copy_from_device(src.data, size)
        else:
            dst.copy_from_device_async(src.data, size, stream)

    def to_device(self, dst, size, offset=0, stream=None):
        src = self.memory + offset
        if stream is None:
            dst.data.copy_from_device(src, size)
        else:
            dst.data.copy_from_device_async(src, size, stream)

    def ptr(self):
        return self.memory.ptr

    def buffer(self, size):
        return ctypes.cast(
            self.memory.ptr,
            ctypes.POINTER(ctypes.c_ubyte * size)
        ).contents

    def array(self, shape, offset=0, dtype=np.float32):
        if dtype is None:
            raise TypeError('dtype must be an instance of numpy.dtype class')
        return cp.ndarray(shape, memptr=self.memory + offset, dtype=dtype)


def extract_params_set_data(model):
    return [param for _, param in sorted(model.namedparams())
            if param.data is not None]


def extract_params_set_grad(model):
    return [param for _, param in sorted(model.namedparams())
            if param.grad is not None]


def pack_params(params, itemsize, attr_name, buffer, stream=None):
    offset = 0
    for param in params:
        v = getattr(param, attr_name)
        size = v.size * itemsize
        buffer.from_device(v, size, offset, stream)
        offset += size


def unpack_params(params, itemsize, attr_name, buffer, stream=None):
    offset = 0
    for param in params:
        v = getattr(param, attr_name)
        size = v.size * itemsize
        buffer.to_device(v, size, offset, stream)
        offset += size

class ParamsData(object):

    def __init__(self, params, attr_name):
        n_params = len(params)
        params_dptr = np.empty(n_params, dtype=np.int64)
        params_dtype = np.empty(n_params, dtype=np.int32)
        params_size_csum = np.empty(n_params+1, dtype=np.int32)
        params_size_csum[0] = 0
        for i, param in enumerate(params):
            v = getattr(param, attr_name)
            params_dptr[i] = v.data.ptr
            params_dtype[i] = _get_nccl_type_id(v.dtype)
            params_size_csum[i+1] = params_size_csum[i] + v.size
        self.n_params = n_params
        self.n_elems = params_size_csum[n_params]
        self.params_size_csum = cp.asarray(params_size_csum)
        self.params_dtype = cp.asarray(params_dtype)
        self.params_dptr = cp.asarray(params_dptr)


def batched_pack_params(params_data, buffer, dtype):
    n_params = params_data.n_params
    n_elems = params_data.n_elems
    params_dptr = params_data.params_dptr
    params_dtype = params_data.params_dtype
    params_size_csum = params_data.params_size_csum
    buf_dtype = _get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    _cupy_batched_pack_params()((n_blocks, ), (n_threads, ),
                                (buffer.memory.ptr, buf_dtype, n_elems,
                                 params_dptr, params_dtype, params_size_csum, n_params))


def batched_unpack_params(params_data, buffer, dtype):
    n_params = params_data.n_params
    n_elems = params_data.n_elems
    params_dptr = params_data.params_dptr
    params_dtype = params_data.params_dtype
    params_size_csum = params_data.params_size_csum
    buf_dtype = _get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    _cupy_batched_unpack_params()((n_blocks, ), (n_threads, ),
                                (buffer.memory.ptr, buf_dtype, n_elems,
                                 params_dptr, params_dtype, params_size_csum, n_params))


def _cupy_batched_pack_params():
    return cp.RawKernel(r'''
#include <cuda_fp16.h>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
    extern "C" __global__
    void cupy_batched_pack_params( void *dst0, int dst_dtype, int n_elems,
                                   unsigned long *params_dptr, int *params_dtype, int *params_size_csum, int n_params) {
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
    ''', 'cupy_batched_pack_params'
    )


def _cupy_batched_unpack_params():
    return cp.RawKernel(r'''
#include <cuda_fp16.h>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
    extern "C" __global__
    void cupy_batched_unpack_params( void *src0, int src_dtype, int n_elems,
                                     unsigned long *params_dptr, int *params_dtype, int *params_size_csum, int n_params) {
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
            else if (src_dtype == NCCL_FLOAT32) {
                ((float*) (params_dptr[j]))[dst_idx] = (float) src[tid];
            }
        }
        else if (src_dtype == NCCL_FLOAT32) {
            float* src = (float*) src0;
            if (dst_dtype == NCCL_FLOAT16) {
                ((half*) (params_dptr[j]))[dst_idx] = (half) src[tid];
            }
            else if (src_dtype == NCCL_FLOAT32) {
                ((float*) (params_dptr[j]))[dst_idx] = (float) src[tid];
            }
       }
    }
    ''', 'cupy_batched_unpack_params'
    )


def _get_nccl_type_id(dtype):
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    else:
        raise ValueError(
            'dtype must be float16 or float32.')


def array_to_buffer_object(array, mpi_dtype=mpi4py.MPI.FLOAT):
    xp = chainer.backend.get_array_module(array)

    if xp is np:
        return get_device_memory_pointer(array)
    else:
        return (get_device_memory_pointer(array), mpi_dtype)


def get_device_memory_pointer(array):
    xp = chainer.backend.get_array_module(array)
    array = xp.ascontiguousarray(array)

    if xp is np:
        return array
    else:
        return ctypes.cast(
            array.data.ptr,
            ctypes.POINTER(ctypes.c_ubyte * array.nbytes)
        ).contents
