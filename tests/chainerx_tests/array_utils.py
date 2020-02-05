import functools
import operator

import numpy

import chainerx


def total_size(shape):
    return functools.reduce(operator.mul, shape, 1)


def uniform(shape, dtype, low=None, high=None, *, random_state=numpy.random):
    kind = numpy.dtype(dtype).kind
    if kind == 'f':
        return (random_state.uniform(
            -1 if low is None else low, 1 if high is None else high, shape)
            .astype(dtype, copy=False))
    if kind == 'u':
        return random_state.randint(
            0 if low is None else low, 4 if high is None else high,
            size=shape, dtype=dtype)
    if kind == 'i':
        return random_state.randint(
            -2 if low is None else low, 3 if high is None else high,
            size=shape, dtype=dtype)
    if kind == 'b':
        return random_state.randint(
            0 if low is None else low, 2 if high is None else high,
            size=shape, dtype=dtype)
    assert False, dtype


def shaped_arange(shape, dtype):
    size = total_size(shape)
    a = numpy.arange(1, size + 1).reshape(shape)
    dtype = numpy.dtype(dtype)
    if dtype == numpy.bool_:
        return a % 2 == 0
    return a.astype(dtype, copy=False)


# TODO(beam2d): Think better way to make multiple different arrays
def create_dummy_ndarray(
        xp, shape, dtype, device=None, pattern=1, padding=True, start=None):
    dtype = chainerx.dtype(dtype).name
    size = total_size(shape)

    if dtype in ('bool', 'bool_'):
        if pattern == 1:
            data = [i % 2 == 1 for i in range(size)]
        else:
            data = [i % 3 == 0 for i in range(size)]
    else:
        if start is None:
            if dtype in chainerx.testing.unsigned_dtypes:
                start = 0 if pattern == 1 else 1
            else:
                start = -1 if pattern == 1 else -2
        data = list(range(start, size + start))

    if padding is True:
        padding = 1
    elif padding is False:
        padding = 0

    # Unpadded array
    a_unpad = numpy.array(data, dtype=dtype).reshape(shape)

    if padding == 0:
        a_np = a_unpad
    else:
        # Create possibly padded (non-contiguous) array.
        # Elements in each axis will be spaced with corresponding padding.
        # The padding for axis `i` is computed as `itemsize * padding[i]`.

        if numpy.isscalar(padding):
            padding = (padding,) * len(shape)
        assert len(padding) == len(shape)

        # Allocate 1-dim raw buffer
        buf_nitems = 1
        for dim, pad in zip((1,) + shape[::-1], padding[::-1] + (0,)):
            buf_nitems = buf_nitems * dim + pad
        # intentionally using uninitialized padding values
        buf_a = numpy.empty((buf_nitems,), dtype=dtype)

        # Compute strides
        strides = []
        st = 1
        itemsize = buf_a.itemsize
        for dim, pad in zip(shape[::-1], padding[::-1]):
            st += pad
            strides.append(st * itemsize)
            st *= dim
        strides = tuple(strides[::-1])

        # Create strided array and copy data
        a_np = numpy.asarray(
            numpy.lib.stride_tricks.as_strided(buf_a, shape, strides))
        a_np[...] = a_unpad

        numpy.testing.assert_array_equal(a_np, a_unpad)

    # Convert to NumPy or chainerx array
    if xp is chainerx:
        a = chainerx.testing._fromnumpy(a_np, keepstrides=True, device=device)
        assert a.strides == a_np.strides
    else:
        a = a_np

    # Checks
    if padding == 0 or all(pad == 0 for pad in padding):
        if xp is chainerx:
            assert a.is_contiguous
        else:
            assert a.flags.c_contiguous
    assert a.shape == shape
    assert a.dtype.name == dtype
    return a


def check_device(a, device=None):
    if device is None:
        device = chainerx.get_default_device()
    elif isinstance(device, str):
        device = chainerx.get_device(device)
    assert a.device is device
