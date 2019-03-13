import numpy


def split_1_11(ary, indices_or_sections, axis=0):
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


class _PatchedNumpy(object):

    def __init__(self):
        np_version = numpy.lib.NumpyVersion(numpy.__version__)
        if np_version < '1.11.0':
            self.split = split_1_11

    def __getattr__(self, name):
        return getattr(numpy, name)


_patched_numpy = _PatchedNumpy()


def _patch_array_module(xp):
    if xp is numpy:
        return _patched_numpy
    else:
        return xp
