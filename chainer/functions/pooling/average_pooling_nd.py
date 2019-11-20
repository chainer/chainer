import functools
import operator

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.pooling import average_pooling_nd_kernel
from chainer.functions.pooling import pooling_nd
from chainer.utils import conv
from chainer.utils import conv_nd
import chainerx


def _get_conv_slices(
        size, k, s, p, cover_all=False, d=1, include_pad=True, dtype='l'):
    """Returns the patch slices.

    Returns:
        A tuple of two 1-D :class:`numpy.ndarrays`\\ s.
        Each represents starting and ending indices of the patches.
    """
    n = conv.get_conv_outsize(size, k, s, p, cover_all, d)
    starts = -p + numpy.arange(n, dtype=dtype) * s
    ends = starts + k
    if not include_pad:
        starts = numpy.maximum(starts, 0)
        ends = numpy.minimum(ends, size)
    return starts, ends


class AveragePoolingND(pooling_nd._PoolingND):

    """Average pooling over a set of N-dimensional planes.

    .. warning::

        This feature is experimental. The interface can change in the future.

    """

    def __init__(
            self, ndim, ksize, stride=None, pad=0, cover_all=False,
            pad_value=0):
        if not (pad_value is None or pad_value == 0):
            raise ValueError(
                'pad_value must be either 0 or None, not {}.'.format(
                    pad_value))

        # TODO(takagi) Support cover_all mode.
        if cover_all is True:
            raise ValueError('`cover_all` mode is not supported yet.')

        super(AveragePoolingND, self).__init__(
            ndim, ksize, stride=stride, pad=pad, cover_all=cover_all)

        self.pad_value = pad_value

    def _get_pooling_width(self, xp, dims, dtype):
        width = None
        for d, k, s, p in six.moves.zip(
                dims, self.ksize, self.stride, self.pad):
            starts, ends = _get_conv_slices(
                d, k, s, p, cover_all=self.cover_all, include_pad=False,
                dtype=dtype)
            w = ends - starts
            if width is None:
                width = w
            else:
                width = numpy.tensordot(width[..., None], w[None, ...], axes=1)
        if xp is cuda.cupy:
            width = cuda.cupy.array(width)
        return width

    def forward_chainerx(self, inputs):
        ndim = self.ndim
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        pad_value = self.pad_value

        x, = inputs
        if x.device.backend.name == 'cuda' and ndim not in (2, 3):
            return chainer.Fallback

        if pad_value == 0:
            pad_mode = 'zero'
        elif pad_value is None:
            pad_mode = 'ignore'
        else:
            assert False

        y = chainerx.average_pool(x, ksize, stride, pad, pad_mode)
        return y,

    def forward_cpu(self, inputs):
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        pad_value = self.pad_value
        cover_all = self.cover_all

        x, = inputs
        in_shape = x.shape
        in_dtype = x.dtype

        col = conv_nd.im2col_nd_cpu(x, ksize, stride, pad, cover_all=cover_all)

        # mean along (_, _, k_1, k_2, ..., k_N, _, ..., _)
        y_axis = tuple(six.moves.range(2, 2 + len(ksize)))
        if pad_value is None:
            dims = x.shape[2:]
            width = self._get_pooling_width(numpy, dims, x.dtype)
            y = col.sum(axis=y_axis) / width
        else:
            assert pad_value == 0
            y = col.mean(axis=y_axis)
            width = None

        self.width = width
        self._in_shape = in_shape
        self._in_dtype = in_dtype
        return y,

    def forward_gpu(self, inputs):
        if chainer.should_use_cudnn('>=auto') and 2 <= self.ndim <= 3:
            # With cuDNN v3 or greater, use cuDNN implementation for inputs
            # with spatial dimensions of two or more.
            return self.forward_cudnn(inputs)

        ndim = self.ndim
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        pad_value = self.pad_value
        cover_all = self.cover_all

        x, = inputs
        in_shape = x.shape
        in_dtype = x.dtype

        n, c = in_shape[:2]
        idims = in_shape[2:]
        odims = tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=cover_all)
            for (d, k, s, p) in six.moves.zip(idims, ksize, stride, pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + odims
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)
        if pad_value is None:
            coeff = self._get_pooling_width(cuda.cupy, idims, x.dtype)
            coeff = cuda.cupy.reciprocal(coeff, out=coeff)
        else:
            assert pad_value == 0
            coeff = 1. / functools.reduce(operator.mul, ksize)

        in_params, out_params, operation, name = \
            average_pooling_nd_kernel.AveragePoolingNDKernelForward.generate(
                ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x.reduced_view(),
            *(idims + odims + ksize + stride + pad + (coeff, y)))

        self.coeff = coeff
        self._in_shape = in_shape
        self._in_dtype = in_dtype
        return y,

    def backward(self, indexes, gy):
        return AveragePoolingNDGrad(self).apply(gy)

    def get_cudnn_pool_mode(self):
        if self.pad_value is None:
            return cuda.cuda.cudnn.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
        else:
            assert self.pad_value == 0
            return cuda.cuda.cudnn.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING


class AveragePoolingNDGrad(function_node.FunctionNode):

    def __init__(self, apoolnd):
        self.func = apoolnd

    def forward_cpu(self, gys):
        func = self.func
        pad_value = func.pad_value
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        in_shape = func._in_shape

        gy, = gys
        idims = in_shape[2:]
        odims = gy.shape[2:]
        colon = slice(None, None, None)
        is_pad_value_none = pad_value is None
        if is_pad_value_none:
            numpy.divide(gy, func.width, out=gy)
        gy_index = (colon, colon) + (None,) * len(idims)
        gcol_reps = (1, 1) + ksize + (1,) * len(odims)
        gcol = numpy.tile(gy[gy_index], gcol_reps)
        gx = conv_nd.col2im_nd_cpu(gcol, stride, pad, idims)
        if not is_pad_value_none:
            gx /= functools.reduce(operator.mul, ksize)
        return gx,

    def forward_gpu(self, gys):
        func = self.func

        if func.is_cudnn_used:
            return func.backward_cudnn(gys)

        ndim = func.ndim
        pad_value = func.pad_value
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        in_shape = func._in_shape
        in_dtype = func._in_dtype

        is_pad_value_none = pad_value is None

        gy, = gys
        n, c = in_shape[:2]
        idims = in_shape[2:]
        odims = gy.shape[2:]
        if is_pad_value_none:
            # This conversion from chainerx to cupy exists here for
            # double backward of chainerx on cuda.
            coeff = backend.from_chx(func.coeff)
            gy *= coeff
        gx = cuda.cupy.empty(in_shape, in_dtype)

        in_params, out_params, operation, name = \
            average_pooling_nd_kernel.AveragePoolingNDKernelBackward.generate(
                ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            gy.reduced_view(),
            *(idims + odims + ksize + stride + pad + (gx,)))

        if not is_pad_value_none:
            gx /= functools.reduce(operator.mul, ksize)
        return gx,

    def backward(self, indexes, grad_outputs):
        func = self.func
        ndim = func.ndim
        pad_value = func.pad_value
        ksize = func.ksize
        stride = func.stride
        pad = func.pad

        return AveragePoolingND(
            ndim, ksize, stride, pad, cover_all=False, pad_value=pad_value
        ).apply(grad_outputs)


def average_pooling_nd(x, ksize, stride=None, pad=0, pad_value=0):
    """N-dimensionally spatial average pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This function provides a N-dimensionally generalized version of
    :func:`~chainer.functions.average_pooling_2d`. This acts similarly to
    :func:`~chainer.functions.convolution_nd`, but it computes the average of
    input spatial patch for each channel without any parameter instead of
    computing the inner products.

    Args:
        x(~chainer.Variable): Input variable.
        ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent. If
            ``None`` is specified, then it uses same stride as the pooling
            window size.
        pad (int or tuple of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        pad_value (0 or None):
            Value to fill the padded region when calculating average.
            If ``None`` is specified, such region is ignored.
            The default value is ``0``, therefore the averages are biased
            towards zero.

    Returns:
        ~chainer.Variable: Output variable.

    .. note::

       This function currently does not support ``cover_all`` mode as
       :func:`max_pooling_nd`. Average pooling runs in non-cover-all mode.

    """
    ndim = len(x.shape[2:])
    return AveragePoolingND(
        ndim, ksize, stride=stride, pad=pad, pad_value=pad_value
    ).apply((x,))[0]


def average_pooling_1d(x, ksize, stride=None, pad=0, pad_value=0):
    """1-dimensional spatial average pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    .. note::

        This function calls :func:`~chainer.functions.average_pooling_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.average_pooling_nd`.

    """
    if len(x.shape[2:]) != 1:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 1. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return average_pooling_nd(x, ksize, stride, pad, pad_value)


def average_pooling_3d(x, ksize, stride=None, pad=0, pad_value=0):
    """3-dimensional spatial average pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    .. note::

        This function calls :func:`~chainer.functions.average_pooling_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.average_pooling_nd`.

    """
    if len(x.shape[2:]) != 3:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 3. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return average_pooling_nd(x, ksize, stride, pad, pad_value)
