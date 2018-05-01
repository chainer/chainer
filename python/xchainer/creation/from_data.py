import numpy

import xchainer


def fromfile(file, dtype=float, count=-1, sep='', device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name
    return xchainer.array(numpy.fromfile(file, dtype, count, sep), device=device)


def loadtxt(
        fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0,
        encoding='bytes', device=None):
    if isinstance(dtype, xchainer.dtype):
        dtype = dtype.name
    return xchainer.array(numpy.loadtxt(
        fname, dtype=dtype, comments=comments, delimiter=delimiter, converters=converters, skiprows=skiprows, usecols=usecols,
        unpack=unpack, ndmin=ndmin, encoding=encoding), device=device)
