# This file defines workaround implementation for
# NumPy-compatibility functions that fall back to NumPy/CuPy functions
# for native/cuda devices respecitvely.
# The workaround does not support backprop, and also requires external
# libraries mentioned above.
# Functions defined in this file should be considered to have high priority for
# genuine implementations.

import numpy

import chainerx


try:
    import cupy
except Exception:
    cupy = None


def _to_numpy(array):
    assert isinstance(array, chainerx.ndarray)
    return chainerx.to_numpy(array, copy=False)


def _from_numpy(array):
    assert isinstance(array, numpy.ndarray)
    return chainerx.array(array, copy=False)


def _to_cupy(array):
    assert cupy is not None
    # Convert to cupy.ndarray on the same device as source array
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


def _from_cupy(array):
    assert cupy is not None
    assert isinstance(array, cupy.ndarray)
    device = chainerx.get_device('cuda', array.device.id)
    return chainerx._core._fromrawpointer(
        array.data.mem.ptr,
        array.shape,
        array.dtype,
        array.strides,
        device,
        array.data.ptr - array.data.mem.ptr,
        array)


def _from_chainerx(array):
    # Converts chainerx.ndarray to numpy/cupy.ndarray.
    # Objects with other types are kept intact.
    if not isinstance(array, chainerx.ndarray):
        return array
    backend_name = array.device.backend.name
    if backend_name == 'native':
        return _to_numpy(array)
    if backend_name == 'cuda':
        if cupy is None:
            raise RuntimeError(
                'ChainerX fallback implementation for cuda backend requires '
                'cupy to be installed.')
        return _to_cupy(array)
    raise RuntimeError(
        'ChainerX fallback implementation only supports native or cuda '
        'backends.')


def _to_chainerx(array):
    # Converts numpy/cupy.ndarray to chainerx.ndarray.
    # Objects with other types are kept intact.
    if isinstance(array, numpy.ndarray):
        return _from_numpy(array)
    elif cupy is not None and isinstance(array, cupy.ndarray):
        return _from_cupy(array)
    return array


def populate():
    ndarray = chainerx.ndarray

    # __getitem__ with advanced indexing
    old_getitem = ndarray.__getitem__

    def __getitem__(arr, key):
        try:
            return old_getitem(arr, key)
        except (IndexError, chainerx.DimensionError):
            pass

        is_backprop_required = arr.is_backprop_required()

        arr = _from_chainerx(arr)
        key = _from_chainerx(key)

        if cupy is not None and isinstance(arr, cupy.ndarray):
            with arr.device:
                ret = arr[key]
        else:
            ret = arr[key]

        # Doing this check after the fallback __getitem__ because the error
        # which caused the fallback might not be due to advanced indexing.
        # In such case the fallback __getitem__ should also raise the error.

        if is_backprop_required:
            raise RuntimeError(
                'ChainerX getitem fallback for advanced indexing is not '
                'supported for arrays that are connected to a graph.')

        return _to_chainerx(ret)

    # __setitem__ with advanced indexing
    def __setitem__(self, key, value):
        if self.is_backprop_required():
            raise RuntimeError(
                'ChainerX setitem fallback for advanced indexing is not '
                'supported for arrays that are connected to a graph.')

        self = _from_chainerx(self)
        key = _from_chainerx(key)
        value = _from_chainerx(value)

        if cupy is not None and isinstance(self, cupy.ndarray):
            with self.device:
                self[key] = value
        else:
            self[key] = value

    ndarray.__setitem__ = __setitem__
    ndarray.__getitem__ = __getitem__
