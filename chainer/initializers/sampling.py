import numpy

from chainer.backends import cuda
from chainer import initializer

# Original code from Berkeley FCN
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py


def _get_linear_filter(size, ndim, upsampling=True):
    """Make a 2D and 3D linear kernel suitable for up/downsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1.
    else:
        center = factor - 0.5
    slices = (slice(size),) * ndim
    og = numpy.ogrid[slices]
    filt = 1.
    for og_i in og:
        filt = filt * (1. - abs(og_i - center) / factor)
    if not upsampling:
        filt /= filt.sum()
    return filt


class _SamplingFilter(initializer.Initializer):

    def __init__(self, upsampling=True, interpolation='linear', dtype=None):
        self._upsampling = upsampling
        if interpolation == 'linear':
            self._get_filter_func = _get_linear_filter
        else:
            raise ValueError(
                'Unsupported interpolation method: {}'.format(interpolation))
        super(_SamplingFilter, self).__init__(dtype)

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
                if ksize != k:
                    raise ValueError(
                        'ksize must be all same: {} != {}'.format(ksize, k))

        filt = self._get_filter_func(
            ksize, ndim=array.ndim - 2, upsampling=self._upsampling)
        filt = xp.asarray(filt)

        array[...] = 0.
        if out_c == 1:
            array[xp.arange(in_c), 0, ...] = filt
        else:
            array[xp.arange(in_c), xp.arange(out_c), ...] = filt


class UpsamplingDeconvFilter(_SamplingFilter):

    """Initializes array with upsampling filter.

    The array is initialized with a standard image upsampling weight.
    This initializer is often used as initial weight for
    :func:`~chainer.links.DeconvolutionND`.
    :func:`~chainer.links.DeconvolutionND` is expected that its `stride` is
    equal to `(ksize + 1) // 2`.

    Reference: Long et al., https://arxiv.org/abs/1411.4038

    Attributes:
        interpolation (str): Upsampling interpolation method.
        Default is 'linear'.

    """

    def __init__(self, interpolation='linear', dtype=None):
        if interpolation != 'linear':
            raise ValueError(
                'Unsupported interpolation method: {}'.format(interpolation))
        super(UpsamplingDeconvFilter, self).__init__(
            upsampling=True, interpolation=interpolation, dtype=dtype)


class DownsamplingConvFilter(_SamplingFilter):

    """Initializes array with downsampling filter.

    The array is initialized with a standard image downsampling weight.
    This initializer is often used as initial weight for
    :func:`~chainer.links.ConvolutionND`.
    :func:`~chainer.links.ConvolutionND` is expected that its `stride` is
    equal to `(ksize + 1) // 2`.

    Reference: Long et al., https://arxiv.org/abs/1411.4038

    Attributes:
        interpolation (str): Downsampling interpolation method.
        Default is 'linear'.

    """

    def __init__(self, interpolation='linear', dtype=None):
        if interpolation != 'linear':
            raise ValueError(
                'Unsupported interpolation method: {}'.format(interpolation))
        super(DownsamplingConvFilter, self).__init__(
            upsampling=False, interpolation=interpolation, dtype=dtype)
