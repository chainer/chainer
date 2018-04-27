import numpy

import xchainer


def fromfile(file, dtype=float, count=-1, sep='', device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name
    return xchainer.array(numpy.fromfile(file, dtype, count, sep), device=device)
