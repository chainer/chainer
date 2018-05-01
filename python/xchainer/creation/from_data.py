import numpy

import xchainer


def fromfile(file, dtype=float, count=-1, sep='', device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name
    return xchainer.array(numpy.fromfile(file, dtype, count, sep), device=device)


# TODO(hvy): Optimize with pre-allocated memory using count for non-native devices.
def fromiter(iterable, dtype, count=-1, device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name
    return xchainer.array(numpy.fromiter(iterable, dtype, count), device=device)
