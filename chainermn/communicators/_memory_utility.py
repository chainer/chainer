import ctypes

import mpi4py.MPI
import numpy as np
from chainermn.communicators import _communication_utility

import chainer.backends
import chainerx as chx

try:
    import cupy as cp
    _cupy_avail = True
except Exception:
    cp = None
    _cupy_avail = False


def _get_memory_pointer_from_chainerx(array):
    # Currently, ChainerMN requires CuPy to support ChainerX.
    # This is because ChainerX's backend does not provide a raw
    # memory pointer class.
    return cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(
            array.data_ptr + array.offset,
            array.data_size,
            array,
            array.device.index),
        0)


class ParamsData(object):
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
            xp = chainer.backend.get_array_module(v)

            if xp == cp:
                v_data = v.data
            elif xp == chx:
                v_data = _get_memory_pointer_from_chainerx(v)
            else:
                raise ValueError(
                    '{} is from an unsupported array module'.format(type(v)))

            params_dptr[i] = v_data.ptr
            if v.dtype not in [np.float16, np.float32, np.float64]:
                raise ValueError('dtype must be float16, float32 or float64.')
            params_dtype[i] = _communication_utility._get_nccl_type_id(v.dtype)
            params_size_csum[i+1] = params_size_csum[i] + v.size
        self.n_params = n_params
        self.n_elems = params_size_csum[n_params]
        self.size_csum = chainer.cuda.cupy.asarray(params_size_csum)
        self.dtype = chainer.cuda.cupy.asarray(params_dtype)
        self.dptr = chainer.cuda.cupy.asarray(params_dptr)


class HostPinnedMemory(object):

    def __init__(self):
        if not _cupy_avail:
            raise RuntimeError('HostPinnedMemory cannot be used: ' +
                               'Cupy is not available.')
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
            raise RuntimeError('DeviceMemory cannot be used: ' +
                               'Cupy is not available.')
        self.size = 0
        self.memory = None

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc(size)

    def from_device(self, src, size, offset=0, stream=None):
        dst = self.memory + offset
        xp = chainer.backend.get_array_module(src)
        if xp == cp:
            src_data = src.data
        elif xp == chx:
            src_data = _get_memory_pointer_from_chainerx(src)
        else:
            raise ValueError(
                '{} is from an unsupported array module'.format(type(src)))
        if stream is None:
            dst.copy_from_device(src_data, size)
        else:
            dst.copy_from_device_async(src_data, size, stream)

    def to_device(self, dst, size, offset=0, stream=None):
        src = self.memory + offset
        xp = chainer.backend.get_array_module(dst)
        if xp == cp:
            dst_data = dst.data
        elif xp == chx:
            dst_data = _get_memory_pointer_from_chainerx(dst)
        else:
            raise ValueError(
                '{} is from an unsupported array module'.format(type(dst)))
        if stream is None:
            dst_data.copy_from_device(src, size)
        else:
            dst_data.copy_from_device_async(src, size, stream)

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


def extract_params_set_grad(model, zero_fill):
    if zero_fill:
        return [param for _, param in sorted(model.namedparams())
                if param.data is not None]
    else:
        return [param for _, param in sorted(model.namedparams())
                if param.data is not None and param.grad is not None]


def count_grad_elements(params, zero_fill):
    if zero_fill:
        return sum(param.data.size for param in params)
    else:
        return sum(param.grad.size for param in params)


def pack_params(params, attr_name, buffer,
                transfer_dtype, zero_fill, stream=None):
    if len(params) == 0:
        return

    # NOTE: dtypes of params might be mixed, in particular f16 & f32.
    offset = 0
    for param in params:
        v = getattr(param, attr_name)
        if attr_name == 'grad' and v is None and zero_fill:
            v = param.xp.zeros_like(param.data)
        size = v.size * np.dtype(transfer_dtype).itemsize
        if v.dtype != transfer_dtype:
            tmp = v.astype(transfer_dtype)
            buffer.from_device(tmp, size, offset, stream)
        else:
            buffer.from_device(v, size, offset, stream)

        offset += size


def unpack_params(params, attr_name, buffer,
                  transfer_dtype, zero_fill, stream=None):
    """Pack parameters into a single CuPy array for efficient communication."""
    if len(params) == 0:
        return
    xp = chainer.backend.get_array_module(getattr(params[0], attr_name))
    offset = 0
    for param in params:
        v = getattr(param, attr_name)
        if attr_name == 'grad' and v is None and zero_fill:
            v = param.xp.empty_like(param.data)
            setattr(param, attr_name, v)
        size = v.size * np.dtype(transfer_dtype).itemsize
        grad_dtype = v.dtype
        if grad_dtype != transfer_dtype:
            v = xp.array(v, copy=False, dtype=transfer_dtype)
        buffer.to_device(v, size, offset, stream)
        offset += size
        if grad_dtype != transfer_dtype:
            # avoid using setattr as ChainerX array cannot be directly updated
            getattr(param, attr_name)[...] = v.astype(grad_dtype)


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
    elif xp is cp:
        return ctypes.cast(
            array.data.ptr,
            ctypes.POINTER(ctypes.c_ubyte * array.nbytes)
        ).contents
    elif xp is chx:
        backend_name = array.device.backend.name
        if backend_name not in ['native', 'cuda']:
            raise ValueError(
                '{} is an unsupported backend'.format(backend_name))
        return ctypes.cast(
            array.data_ptr,
            ctypes.POINTER(ctypes.c_ubyte * array.nbytes)
        ).contents
    else:
        raise ValueError(
            '{} is from an unsupported array module'.format(type(array)))


def _batched_pack_params(params_data, buffer, dtype, stream=None):
    n_params = params_data.n_params
    n_elems = params_data.n_elems
    params_dptr = params_data.dptr
    params_dtype = params_data.dtype
    params_size_csum = params_data.size_csum
    buf_dtype = _communication_utility._get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    if stream is None:
        stream = cp.cuda.get_current_stream()
    with stream:
        _cupy_batched_pack_params()(
            (n_blocks, ), (n_threads, ),
            (buffer.memory.ptr, buf_dtype, n_elems,
             params_dptr, params_dtype, params_size_csum, n_params))


def _batched_unpack_params(params_data, buffer, dtype, stream=None):
    n_params = params_data.n_params
    n_elems = params_data.n_elems
    params_dptr = params_data.dptr
    params_dtype = params_data.dtype
    params_size_csum = params_data.size_csum
    buf_dtype = _communication_utility._get_nccl_type_id(dtype)
    n_threads = 128
    n_blocks = (n_elems + n_threads - 1) // n_threads
    if stream is None:
        stream = cp.cuda.get_current_stream()
    with stream:
        _cupy_batched_unpack_params()(
            (n_blocks, ), (n_threads, ),
            (buffer.memory.ptr, buf_dtype, n_elems,
             params_dptr, params_dtype, params_size_csum, n_params))


def _cupy_batched_pack_params():
    return chainer.cuda.raw(r'''
#include <cupy/carray.cuh>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
#define NCCL_FLOAT64  8
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
            else if (src_dtype == NCCL_FLOAT64) {
                dst[tid] = (half) (((double*) (params_dptr[j]))[src_idx]);
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
            else if (src_dtype == NCCL_FLOAT64) {
                dst[tid] = (float) (((double*) (params_dptr[j]))[src_idx]);
            }
       }
       else if (dst_dtype == NCCL_FLOAT64) {
            double* dst = (double*) dst0;
            if (src_dtype == NCCL_FLOAT16) {
                dst[tid] = (double) (((half*) (params_dptr[j]))[src_idx]);
            }
            else if (src_dtype == NCCL_FLOAT32) {
                dst[tid] = (double) (((float*) (params_dptr[j]))[src_idx]);
            }
            else if (src_dtype == NCCL_FLOAT64) {
                dst[tid] = (double) (((double*) (params_dptr[j]))[src_idx]);
            }
       }
    }
    ''', 'cupy_batched_pack_params')


def _cupy_batched_unpack_params():
    return chainer.cuda.raw(r'''
#include <cupy/carray.cuh>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
#define NCCL_FLOAT64  8
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
            else if (dst_dtype == NCCL_FLOAT64) {
                ((double*) (params_dptr[j]))[dst_idx] = (double) src[tid];
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
            else if (dst_dtype == NCCL_FLOAT64) {
                ((double*) (params_dptr[j]))[dst_idx] = (double) src[tid];
            }
       }
       else if (src_dtype == NCCL_FLOAT64) {
            double* src = (double*) src0;
            if (dst_dtype == NCCL_FLOAT16) {
                ((half*) (params_dptr[j]))[dst_idx] = (half) src[tid];
            }
            else if (dst_dtype == NCCL_FLOAT32) {
                ((float*) (params_dptr[j]))[dst_idx] = (float) src[tid];
            }
            else if (dst_dtype == NCCL_FLOAT64) {
                ((double*) (params_dptr[j]))[dst_idx] = (double) src[tid];
            }
       }
    }''', 'cupy_batched_unpack_params')
