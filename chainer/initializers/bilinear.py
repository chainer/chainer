import numpy

from chainer.backends import cuda
from chainer import initializer

# Original code from Berkeley FCN
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py


def _get_upsampling_filter(size, ndim):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1.
    else:
        center = factor - 0.5
    slices = (slice(None, size),) * ndim
    og = numpy.ogrid[slices]
    filt = 1
    for og_i in og:
        filt = filt * (1. - abs(og_i - center) / factor)
    return filt


class Bilinear(initializer.Initializer):
    """Initializes array with bilinear upsampling.

    The array is initialized with a standard image bilinear upsampling weight.
    This initializer is often used as inital weight for
    :func:`~chainer.links.Deconvolution2D`.

    Reference: Long et al., https://arxiv.org/abs/1411.4038

    """

    def __init__(self, dtype=None):
        super(Bilinear, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = cuda.get_array_module(array)

        in_c, out_c = array.shape[:2]
        assert in_c == out_c or out_c == 1

        ksize = None
        for k in array.shape[2:]:
            if ksize is None:
                ksize = k
            else:
                assert ksize == k

        filt = _get_upsampling_filter(ksize, ndim=array.ndim - 2)
        filt = xp.asarray(filt)

        array[...] = 0.
        if out_c == 1:
            array[range(in_c), 0, ...] = filt
        else:
            array[range(in_c), range(out_c), ...] = filt
