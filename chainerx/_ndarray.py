# This file defines inefficient workaround implementation for
# NumPy ndarray-compatibility functions. This file should ultimately be emptied by
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


def _to_numpy(array):
    assert isinstance(array, chainerx.ndarray)
    return chainerx.to_numpy(array, copy=False)


def _from_numpy(array):
    assert isinstance(array, numpy.ndarray)
    return chainerx.array(array, copy=False)


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


def _from_cupy(array):
    if cupy is None:
        raise RuntimeError(
            'Currently cupy is required in this operation.')
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


# Populates chainerx.ndarray methods in the chainerx namespace
def populate():
    ndarray = chainerx.ndarray

    old_getitem = ndarray.__getitem__

    def __getitem__(self, key):
        """Returns self[key].

        Supports both basic and advanced indexing.
        """
        try:
            return old_getitem(self, key)
        except (IndexError, chainerx.DimensionError) as e:
            pass

        # fallback
        if self.device.backend.name == 'native':
            if isinstance(key, chainerx.ndarray):
                key = _to_numpy(key)
            return _from_numpy(_to_numpy(self).__getitem__(key))
        elif self.device.backend.name == 'cuda':
            if isinstance(key, chainerx.ndarray):
                key = _to_cupy(key)
            return _from_cupy(_to_cupy(self).__getitem__(key))
        else:
            raise NotImplementedError(
                'Currently __getitem__ fallback is supported only in '
                'native and cuda backend.')

    def __setitem__(self, key, value):
        """Sets self[key] to value.

        Supports both basic and advanced indexing.

        Note:

            With the ``cuda`` backend, the behavior differs from NumPy when
            integer arrays in ``slices`` reference the same location
            multiple times. In that case, the value that is actually stored
            is undefined.

            >>> import chainerx
            >>> chainerx.set_default_device('cuda:0')
            >>> a = chainerx.zeros((2,), dtype=chainerx.float)
            >>> i = chainerx.array([0, 1, 0, 1, 0, 1])
            >>> v = chainerx.arange(6).astype(chainerx.float)
            >>> a[i] = v
            >>> a  # doctest: +SKIP
            array([2., 3.], shape=(2,), dtype=float64, device='cuda:0')

            On the other hand, NumPy and ``native`` backend store the value
            corresponding to the last index among the indices referencing
            duplicate locations.

            >>> import numpy
            >>> a_cpu = numpy.zeros((2,), dtype=numpy.float)
            >>> i_cpu = numpy.array([0, 1, 0, 1, 0, 1])
            >>> v_cpu = numpy.arange(6).astype(numpy.float)
            >>> a_cpu[i_cpu] = v_cpu
            >>> a_cpu
            array([4., 5.])

        """
        if self.device.backend.name == 'native':
            if isinstance(value, ndarray):
                value = _to_numpy(value)
            if isinstance(key, ndarray):
                key = _to_numpy(key)
            _to_numpy(self).__setitem__(key, value)
        elif self.device.backend.name == 'cuda':
            # Convert to cupy.ndarray on the same device as source array
            if isinstance(value, ndarray):
                value = _to_cupy(value)
            if isinstance(key, ndarray):
                key = _to_cupy(key)
            self_cupy = _to_cupy(self)
            with self_cupy.device:
                self_cupy.__setitem__(key, value)
        else:
            raise NotImplementedError(
                'Currently item assignment is supported only in native and '
                'cuda backend.')

    def clip(self, a_min, a_max):
        """Returns an array with values limited to [``a_min``, ``a_max``].

        .. seealso:: :func:`chainerx.clip` for full documentation,
            :meth:`numpy.ndarray.clip`

        """
        return chainerx.clip(self, a_min, a_max)

    def ravel(self):
        """Returns an array flattened into one dimension.

        .. seealso:: :func:`chainerx.ravel` for full documentation,
            :meth:`numpy.ndarray.ravel`

        """
        return chainerx.ravel(self)

    ndarray.__setitem__ = __setitem__
    ndarray.__getitem__ = __getitem__
    ndarray.clip = clip
    ndarray.ravel = ravel
