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


def populate():
    # Populates workaround functions in the chainerx namespace
    _populate_chainerx()
    _populate_ndarray()
    _populate_random()


def _populate_chainerx():
    # Populates chainerx toplevel functions

    def square(x):
        """Square of the input, element-wise.

        Args:
            x (~chainerx.ndarray or scalar): Input data

        Returns:
            ~chainerx.ndarray: Returned array: :math:`y = x * x`.
            A scalar is returned if ``x`` is a scalar.

        Note:
            During backpropagation, this function propagates the gradient
            of the output array to the input array ``x``.

        .. seealso:: :func:`numpy.square`
        """
        return x * x

    def clip(a, a_min, a_max):
        """Clips the values of an array to a given interval.

        Given an interval, values outside the interval are clipped to the
        interval edges. For example, if an interval of ``[0, 1]`` is specified,
        values smaller than 0 become 0, and values larger than 1 become 1.

        Args:
            a (~chainerx.ndarray): Array containing elements to clip.
            a_min (scalar): Maximum value.
            a_max (scalar): Minimum value.

        Returns:
            ~chainerx.ndarray: An array with the elements of ``a``, but where
            values < ``a_min`` are replaced with ``a_min``,
            and those > ``a_max`` with ``a_max``.

        Note:
            The :class:`~chainerx.ndarray` typed ``a_min` and ``a_max`` are
            not supported yet.

        Note:
            During backpropagation, this function propagates the gradient
            of the output array to the input array ``a``.

        .. seealso:: :func:`numpy.clip`
        """
        return -chainerx.maximum(-chainerx.maximum(a, a_min), -a_max)

    def ravel(a):
        """Returns a flattened array.

        It tries to return a view if possible, otherwise returns a copy.

        Args:
            a (~chainerx.ndarray): Array to be flattened.

        Returns:
            ~chainerx.ndarray: A flattened view of ``a`` if possible,
            otherwise a copy.

        Note:
            During backpropagation, this function propagates the gradient
            of the output array to the input array ``a``.

        .. seealso:: :func:`numpy.ravel`
        """
        return a.reshape((a.size,))

    chainerx.square = square
    chainerx.clip = clip
    chainerx.ravel = ravel


def _populate_ndarray():
    # Populates chainerx.ndarray methods
    ndarray = chainerx.ndarray

    old_getitem = ndarray.__getitem__

    def __getitem__(self, key):
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
        if self.device.backend.name == 'native':
            if isinstance(value, ndarray):
                value = _to_numpy(value)
            _to_numpy(self).__setitem__(key, value)
        elif self.device.backend.name == 'cuda':
            # Convert to cupy.ndarray on the same device as source array
            if isinstance(value, ndarray):
                value = _to_cupy(value)
            self_cupy = _to_cupy(self)
            with self_cupy.device:
                self_cupy.__setitem__(key, value)
        else:
            raise NotImplementedError(
                'Currently item assignment is supported only in native and '
                'cuda backend.')

    def clip(self, a_min, a_max):
        """Returns an array with values limited to [``a_min``, ``a_max``].

        ... seealso: :func:`chainerx.clip` for full documentation,
        :func:`numpy.ndarray.clip`
        """
        return chainerx.clip(self, a_min, a_max)

    def ravel(self):
        """Returns an array flattened into one dimension.

        ... seealso: :func:`chainerx.ravel` for full documentation,
        :func:`numpy.ndarray.ravel`
        """
        return chainerx.ravel(self)

    ndarray.__setitem__ = __setitem__
    ndarray.__getitem__ = __getitem__
    ndarray.clip = clip
    ndarray.ravel = ravel


def _populate_random():
    # Populates chainerx.random package

    def normal(*args, device=None, **kwargs):
        """Draws random samples from a normal (Gaussian) distribution.

        This is currently equivalent to :func:`numpy.random.normal`
        wrapped by :func:`chainerx.array`, given the device argument.

        .. seealso:: :func:`numpy.random.normal`
        """
        a = numpy.random.normal(*args, **kwargs)
        return chainerx.array(a, device=device, copy=False)

    def uniform(*args, device=None, **kwargs):
        """Draws samples from a uniform distribution.

        This is currently equivalent to :func:`numpy.random.normal`
        wrapped by :func:`chainerx.array`, given the device argument.

        .. seealso:: :func:`numpy.random.normal`
        """
        a = numpy.random.uniform(*args, **kwargs)
        return chainerx.array(a, device=device, copy=False)

    random_ = types.ModuleType('random')
    random_.__dict__['normal'] = normal
    random_.__dict__['uniform'] = uniform
    sys.modules['chainerx.random'] = random_
    chainerx.random = random_
