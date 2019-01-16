import numpy


_np_version = numpy.lib.NumpyVersion(numpy.__version__)


if _np_version >= '1.11.0':
    split = numpy.split
else:
    def split(ary, indices_or_sections, axis=0):
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
