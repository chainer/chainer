import numpy


_np_version = numpy.lib.NumpyVersion(numpy.__version__)


def broadcast_to(xp):
    return _np_1_10_broadcast_to if xp is numpy else xp.broadcast_to


def _np_1_10_broadcast_to(array, shape, subok=False):
    dummy = numpy.empty(shape, dtype=numpy.int8)
    return numpy.broadcast_arrays(array, dummy)[0]


def split(xp):
    return _np_split if xp is numpy else xp.split


def _np_1_11_split(ary, indices_or_sections, axis=0):
    x = ary
    ys = numpy.split(x)
    if all(y.ndim == x.ndim for y in ys):
        return ys
    tmp = [len(t) for t in numpy.split(
        numpy.empty(x.shape[axis], dtype=numpy.int8),
        indices_or_sections, 0)]
    shape = list(x.shape)
    for i, t in enumerate(tmp):
        y = ys[i]
        if y.ndim != x.ndim:
            assert y.size == 0
            shape[axis] = t
            ys[i] = y.reshape(shape)
    return ys


_np_split = numpy.split if _np_version >= '1.11.0' else _np_1_11_split


def sqrt(xp):
    return _np_sqrt if xp is numpy else xp.sqrt


def _np_1_11_2_sqrt(x, out=None, **kwargs):
    # Before NumPy 1.11.2, `numpy.sqrt` casts float16 to float32
    # Note: This func is not a ufunc while numpy.sqrt is.
    if x.dtype == numpy.float16:
        kwargs.setdefault('dtype', numpy.float16)
    return numpy.sqrt(x, out, **kwargs)


_np_sqrt = numpy.sqrt if _np_version >= '1.11.2' else _np_1_11_2_sqrt
