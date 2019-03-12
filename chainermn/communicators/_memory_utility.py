import ctypes

import mpi4py.MPI
import numpy as np

import chainer.backends
try:
    import cupy as cp
    _cupy_avail = True
except ImportError:
    _cupy_avail = False


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
