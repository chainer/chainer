import numpy

import xchainer


def fromfile(file, dtype=float, count=-1, sep='', device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name
    return xchainer.array(numpy.fromfile(file, dtype, count, sep), device=device)


def fromstring(string, dtype=float, count=-1, sep='', device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name

    # sep should always be specified in numpy.fromstring since its default argument has been deprecated since 1.14.
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fromstring.html
    return xchainer.array(numpy.fromstring(string, dtype, count, sep), device=device)
