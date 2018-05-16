import functools
import operator

import xchainer


def total_size(shape):
    return functools.reduce(operator.mul, shape, 1)


def create_dummy_ndarray(xp, shape, dtype, device=None):
    if xchainer.dtype(dtype).name in xchainer.testing.unsigned_dtypes:
        start = 0
        stop = total_size(shape)
    else:
        start = -1
        stop = total_size(shape) - 1

    if xp is xchainer:
        return xp.arange(start=start, stop=stop, device=device).reshape(shape).astype(dtype)
    else:
        return xp.arange(start=start, stop=stop).reshape(shape).astype(dtype)
