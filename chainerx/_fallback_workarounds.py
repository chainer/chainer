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


class _DummyContext:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


_dummy_context = _DummyContext()


def _to_numpy(array):
    assert isinstance(array, chainerx.ndarray)
    return chainerx.to_numpy(array, copy=False)


def _from_numpy(array):
    assert isinstance(array, numpy.ndarray)
    return chainerx.array(array, copy=False)


def _to_cupy(array):
    assert cupy is not None
    # Convert to cupy.ndarray on the same device as source array
    return chainerx._to_cupy(array)


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


def _from_chx(array, check_backprop=True):
    # Converts chainerx.ndarray to numpy/cupy.ndarray.
    # Objects with other types are kept intact.
    # Returns a pair: (xp, cupy device or dummy context, numpy/cupy.ndarray).
    if not isinstance(array, chainerx.ndarray):
        return None, _dummy_context, array
    if check_backprop and array.is_backprop_required():
        raise RuntimeError(
            'ChainerX function fallback using NumPy/CuPy is not '
            'supported for arrays that are connected to a graph.')
    backend_name = array.device.backend.name
    if backend_name == 'native':
        return numpy, _dummy_context, _to_numpy(array)
    if backend_name == 'cuda':
        if cupy is None:
            raise RuntimeError(
                'ChainerX fallback implementation for cuda backend requires '
                'cupy to be installed.')
        array_cupy = _to_cupy(array)
        return cupy, array_cupy.device, array_cupy
    raise RuntimeError(
        'ChainerX fallback implementation only supports native or cuda '
        'backends.')


def _to_chx(array):
    # Converts numpy/cupy.ndarray to chainerx.ndarray.
    # Objects with other types are kept intact.
    if isinstance(array, numpy.ndarray):
        return _from_numpy(array)
    elif cupy is not None and isinstance(array, cupy.ndarray):
        return _from_cupy(array)
    return array


def _populate_module_functions():

    def _hstack(arrs):
        assert len(arrs) > 0
        arrs2 = []
        for a in arrs:
            xp, dev, a2 = _from_chx(a)
            arrs2.append(a2)
        with dev:
            ret = xp.hstack(arrs2)
        return _to_chx(ret)

    chainerx.hstack = _hstack

    def _vstack(arrs):
        assert len(arrs) > 0
        arrs2 = []
        for a in arrs:
            xp, dev, a2 = _from_chx(a)
            arrs2.append(a2)
        with dev:
            ret = xp.vstack(arrs2)
        return _to_chx(ret)

    chainerx.vstack = _vstack

    def _fix(arr):
        xp, dev, arr = _from_chx(arr)
        with dev:
            ret = xp.fix(arr)
            ret = xp.asarray(ret)
        return _to_chx(ret)

    chainerx.fix = _fix


def _populate_ndarray():
    ndarray = chainerx.ndarray

    # __getitem__ with advanced indexing
    old_getitem = ndarray.__getitem__

    def __getitem__(arr, key):
        try:
            return old_getitem(arr, key)
        except (IndexError, chainerx.DimensionError):
            pass

        is_backprop_required = arr.is_backprop_required()

        xp, dev, arr = _from_chx(arr, check_backprop=False)
        if isinstance(key, tuple):
            key = tuple([_from_chx(k, check_backprop=False)[2] for k in key])
        else:
            _, _, key = _from_chx(key, check_backprop=False)

        with dev:
            ret = arr[key]

        # Doing this check after the fallback __getitem__ because the error
        # which caused the fallback might not be due to advanced indexing.
        # In such case the fallback __getitem__ should also raise the error.

        if is_backprop_required:
            raise RuntimeError(
                'ChainerX getitem fallback for advanced indexing is not '
                'supported for arrays that are connected to a graph.')

        return _to_chx(ret)

    # __setitem__ with advanced indexing
    def __setitem__(self, key, value):
        if self.is_backprop_required():
            raise RuntimeError(
                'ChainerX setitem fallback for advanced indexing is not '
                'supported for arrays that are connected to a graph.')

        xp, dev, self = _from_chx(self)
        if isinstance(key, tuple):
            key = tuple([_from_chx(k)[2] for k in key])
        else:
            _, _, key = _from_chx(key)
        _, _, value = _from_chx(value)

        with dev:
            self[key] = value

    ndarray.__setitem__ = __setitem__
    ndarray.__getitem__ = __getitem__

    def tolist(arr):
        _, dev, arr = _from_chx(arr)
        with dev:
            ret = arr.tolist()
        return ret

    ndarray.tolist = tolist


def populate():
    _populate_module_functions()
    _populate_ndarray()
