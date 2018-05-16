import functools
import operator

import xchainer


def total_size(shape):
    return functools.reduce(operator.mul, shape, 1)


# TODO(beam2d): Think better way to make multiple different arrays
def create_dummy_ndarray(xp, shape, dtype, device=None, pattern=1):
    dtype = xchainer.dtype(dtype).name
    size = total_size(shape)
    if pattern == 1:
        if dtype in ('bool', 'bool_'):
            data = [i % 2 == 1 for i in range(size)]
        elif dtype in xchainer.testing.unsigned_dtypes:
            data = list(range(size))
        else:
            data = list(range(-1, size - 1))
    else:
        if dtype in ('bool', 'bool_'):
            data = [i % 3 == 0 for i in range(size)]
        elif dtype in xchainer.testing.unsigned_dtypes:
            data = list(range(1, size + 1))
        else:
            data = list(range(-2, size - 2))

    if xp is xchainer:
        return xp.array(data, dtype=dtype, device=device).reshape(shape)
    else:
        return xp.array(data, dtype=dtype).reshape(shape)
