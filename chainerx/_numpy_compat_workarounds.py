# This file defines inefficient workaround implementation for
# NumPy-compatibility functions. This file should ultimately be emptied by
# implementing those functions in more efficient manner.

import sys
import types

import numpy

import chainerx


# TODO(niboshi): Do not depend on CuPy
try:
    import cupy
except Exception:
    cupy = None


def _to_cupy(array):
    # Convert to cupy.ndarray on the same device as source array
    if cupy is None:
        raise RuntimeError(
            'Currently cupy is required in this operation.')
    return cupy.ndarray(
        array.shape,
        array.dtype,
        cupy.cuda.MemoryPointer(
            cupy.cuda.UnownedMemory(
                array.data_ptr + array.offset,
                array.data_size,
                array,
                array.device.index),
            0),
        strides=array.strides)


def populate():
    # Populates workaround functions in the chainerx namespace
    _populate_chainerx()
    _populate_ndarray()
    _populate_random()


def _populate_chainerx():
    # Populates chainerx toplevel functions

    def square(x):
        return x * x

    chainerx.square = square


def _populate_ndarray():
    # Populates chainerx.ndarray methods
    ndarray = chainerx.ndarray

    def __setitem__(self, key, value):
        if self.device.backend.name == 'native':
            if isinstance(value, chainerx.ndarray):
                value = chainerx.to_numpy(value, copy=False)
            chainerx.to_numpy(self, copy=False).__setitem__(key, value)
        elif self.device.backend.name == 'cuda':
            # Convert to cupy.ndarray on the same device as source array
            if isinstance(value, chainerx.ndarray):
                value = _to_cupy(value)
            self_cupy = _to_cupy(self)
            with self_cupy.device:
                self_cupy.__setitem__(key, value)
        else:
            raise NotImplementedError(
                'Currently item assignment is supported only in native and '
                'cuda backend.')

    def clip(self, a_min, a_max):
        return -chainerx.maximum(-chainerx.maximum(self, a_min), -a_max)

    def ravel(self):
        return self.reshape((self.size,))

    ndarray.__setitem__ = __setitem__
    ndarray.clip = clip
    ndarray.ravel = ravel


def _populate_random():
    # Populates chainerx.random package

    def normal(*args, device=None, **kwargs):
        a = numpy.random.normal(*args, **kwargs)
        return chainerx.array(a, device=device, copy=False)

    random_ = types.ModuleType('random')
    random_.__dict__['normal'] = normal
    sys.modules['chainerx.random'] = random_
    chainerx.random = random_
