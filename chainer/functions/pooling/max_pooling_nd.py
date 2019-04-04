import functools
from operator import mul

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.pooling import max_pooling_nd_kernel
from chainer.functions.pooling import pooling_nd
from chainer.utils import conv_nd
import chainerx


class MaxPoolingND(pooling_nd._PoolingND):

    """Max pooling over a set of N-dimensional planes.

    .. warning::

        This feature is experimental. The interface can change in the future.

    """

    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=True,
                 return_indices=False):
        super(MaxPoolingND, self).__init__(
            ndim, ksize, stride=stride, pad=pad, cover_all=cover_all,
            return_indices=return_indices)

    def forward_chainerx(self, x):
        # TODO(sonots): Support return_indices in ChainerX
        if self.return_indices:
            return chainer.Fallback
        if x[0].device.backend.name == 'cuda':
            # TODO(sonots): Support more ndim in ChainerX
            if self.ndim not in [2, 3]:
                return chainer.Fallback
        return chainerx.max_pool(x[0], self.ksize, self.stride, self.pad,
                                 self.cover_all),

    def forward_cpu(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        col = conv_nd.im2col_nd_cpu(
            x[0], self.ksize, self.stride, self.pad, pval=-float('inf'),
            cover_all=self.cover_all)
        n, c = col.shape[:2]
        mid = (len(col.shape) - 2) // 2 + 2
        ksize = col.shape[2:mid]
        outs = col.shape[mid:]
        # (n, c, k_1 * k_2 * ... * k_N, out_1, out_2, ..., out_N)
        col_shape = (n, c) + (functools.reduce(mul, ksize),) + outs
        col = col.reshape(col_shape)

        # We select maximum twice, since the implementation using numpy.choose
        # hits its bug when kh * kw >= 32.
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y,

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('>=auto') and 2 <= self.ndim <= 3:
            # With cuDNN v3 or greater, use cuDNN implementation for inputs
            # with spatial dimensions of two or more.
            return super(MaxPoolingND, self).forward_gpu(x)

        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = tuple(conv_nd.get_conv_outsize(d, k, s, p, self.cover_all)
                   for (d, k, s, p) in six.moves.zip(
                       dims, self.ksize, self.stride, self.pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x[0].dtype)
        self.indexes = cuda.cupy.empty(y_shape, dtype=numpy.int32)

        in_params, out_params, operation, name = \
            max_pooling_nd_kernel.MaxPoolingNDKernelForward.generate(self.ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x[0].reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad +
              (y, self.indexes)))

        return y,

    def backward(self, indexes, gy):
        return MaxPoolingNDGrad(self).apply(gy)

    def _get_pool_mode(self):
        return cuda.cuda.cudnn.CUDNN_POOLING_MAX


class MaxPoolingNDGrad(function_node.FunctionNode):

    def __init__(self, mpoolnd):
        self.ndim = mpoolnd.ndim
        self.ksize = mpoolnd.ksize
        self.stride = mpoolnd.stride
        self.pad = mpoolnd.pad
        self.cover_all = mpoolnd.cover_all
        self._used_cudnn = mpoolnd._used_cudnn
        if not self._used_cudnn:
            self.indexes = mpoolnd.indexes
            self._in_shape = mpoolnd._in_shape
            self._in_dtype = mpoolnd._in_dtype
        self.mpoolnd = mpoolnd

    def forward_cpu(self, gy):
        ndim = self.ndim
        n, c = gy[0].shape[:2]
        outs = gy[0].shape[2:]
        dims = self._in_shape[2:]
        prod_outs = functools.reduce(mul, outs)
        prod_ksize = functools.reduce(mul, self.ksize)

        gcol = numpy.zeros(
            n * c * prod_outs * prod_ksize, dtype=self._in_dtype)

        indexes = self.indexes.flatten()
        indexes += numpy.arange(0, indexes.size * prod_ksize, prod_ksize)

        gcol[indexes] = gy[0].ravel()
        gcol_shape = (n, c) + outs + self.ksize
        gcol = gcol.reshape(gcol_shape)
        for i in six.moves.range(ndim):
            gcol = numpy.swapaxes(gcol, 2 + i, ndim + 2 + i)

        gx = conv_nd.col2im_nd_cpu(gcol, self.stride, self.pad, dims)
        return gx,

    def forward_gpu(self, gy):
        if self._used_cudnn:
            x = self.mpoolnd.get_retained_inputs()[0].array
            return self.mpoolnd.backward_gpu((x,), gy)

        n, c = self._in_shape[:2]
        dims = self._in_shape[2:]
        ys = gy[0].shape[2:]
        gx = cuda.cupy.empty(self._in_shape, self._in_dtype)

        ndim = self.ndim
        in_params, out_params, operation, name = \
            max_pooling_nd_kernel.MaxPoolingNDKernelBackward.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            gy[0].reduced_view(), self.indexes.reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad + (gx,)))
        return gx,

    def backward(self, indexes, ggx):
        return MaxPoolingNDWithIndexes(self.mpoolnd).apply(ggx)


class MaxPoolingNDWithIndexes(function_node.FunctionNode):

    def __init__(self, mpoolnd):
        self.ndim = mpoolnd.ndim
        self.ksize = mpoolnd.ksize
        self.stride = mpoolnd.stride
        self.pad = mpoolnd.pad
        self.cover_all = mpoolnd.cover_all
        self._used_cudnn = mpoolnd._used_cudnn
        if not self._used_cudnn:
            self.indexes = mpoolnd.indexes
        else:
            self.mpoolnd = mpoolnd

    def forward_cpu(self, x):
        col = conv_nd.im2col_nd_cpu(
            x[0], self.ksize, self.stride, self.pad, pval=-float('inf'),
            cover_all=self.cover_all)
        n, c = col.shape[:2]
        mid = (len(col.shape) - 2) // 2 + 2
        ksize = col.shape[2:mid]
        outs = col.shape[mid:]
        # (n, c, k_1 * k_2 * ... * k_N, out_1, out_2, ..., out_N)
        ksize_total = functools.reduce(mul, ksize)
        col_shape = (n, c) + (ksize_total,) + outs
        col = col.reshape(col_shape)
        # (n, c, out_1, ..., out_N, k_1 * .. * k_N)
        col_indexes = (0, 1) + tuple(six.moves.range(3, 3 + self.ndim)) + (2,)
        col = col.transpose(col_indexes)
        col = col.reshape(-1, ksize_total)

        indexes = self.indexes.ravel()
        col = col[numpy.arange(len(indexes)), indexes]
        return col.reshape((n, c) + outs),

    def forward_gpu(self, inputs):
        if self._used_cudnn:
            x = self.mpoolnd.get_retained_inputs()[0].array
            return self._forward_gpu_compute_indexes_again((x, inputs[0]))
        x, = inputs
        self._in_shape = x.shape
        self._in_dtype = x.dtype

        n, c = x.shape[:2]
        dims = x.shape[2:]

        ys = tuple(conv_nd.get_conv_outsize(d, k, s, p, self.cover_all)
                   for (d, k, s, p) in six.moves.zip(
                       dims, self.ksize, self.stride, self.pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)

        cls = max_pooling_nd_kernel.MaxPoolingNDKernelForwardWithIndexes
        in_params, out_params, operation, name = cls.generate(self.ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x.reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad +
              (self.indexes.reduced_view(), y)))
        return y,

    def _forward_gpu_compute_indexes_again(self, inputs):
        x, ggx = inputs
        self._in_shape = x.shape
        self._in_dtype = x.dtype

        n, c = x.shape[:2]
        dims = x.shape[2:]

        ys = tuple(conv_nd.get_conv_outsize(d, k, s, p, self.cover_all)
                   for (d, k, s, p) in six.moves.zip(
                       dims, self.ksize, self.stride, self.pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)

        cls = max_pooling_nd_kernel.MaxPoolingNDKernelForwardWithIndexes1
        in_params, out_params, operation, name = cls.generate(self.ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x.reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad +
              (ggx.reduced_view(), y)))
        return y,


def max_pooling_nd(x, ksize, stride=None, pad=0, cover_all=True,
                   return_indices=False):
    """N-dimensionally spatial max pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This function provides a N-dimensionally generalized version of
    :func:`~chainer.functions.max_pooling_2d`. This acts similarly to
    :func:`~chainer.functions.convolution_nd`, but it computes the maximum of
    input spatial patch for each channel without any parameter instead of
    computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s,s, ..., s)`` are equivalent. If
            ``None`` is specified, then it uses same stride as the pooling
            window size.
        pad (int or tuple of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        cover_all (bool): If ``True``, all spatial locations are pooled into
            some output pixels. It may make the output size larger.
        return_indices (bool): If ``True``, pooling indices array is returned
            together with the output variable. The returned indices are
            expected for use by :func:`chainer.functions.upsampling_nd`.
            Note that cuDNN will not be used for this function if
            ``return_indices`` is set to ``True``, as cuDNN does not return
            indices information.

    Returns:
        ~chainer.Variable or tuple:
            When ``return_indices`` is ``False`` (default), returns the output
            variable.
            When ``True``, returns the tuple of the output variable and
            pooling indices (:ref:`ndarray`). Pooling indices will be on the
            same device as the input.

    """
    ndim = len(x.shape[2:])

    func = MaxPoolingND(ndim, ksize, stride, pad, cover_all, return_indices)
    if return_indices:
        with chainer.using_config('use_cudnn', 'never'):
            out = func.apply((x,))[0]
        return out, func.indexes

    return func.apply((x,))[0]


def max_pooling_1d(x, ksize, stride=None, pad=0, cover_all=True,
                   return_indices=False):
    """1-dimensional spatial max pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    .. note::

        This function calls :func:`~chainer.functions.max_pooling_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.max_pooling_nd`.

    """
    if len(x.shape[2:]) != 1:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 1. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return max_pooling_nd(x, ksize, stride, pad, cover_all, return_indices)


def max_pooling_3d(x, ksize, stride=None, pad=0, cover_all=True,
                   return_indices=False):
    """3-dimensional spatial max pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    .. note::

        This function calls :func:`~chainer.functions.max_pooling_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.max_pooling_nd`.

    """
    if len(x.shape[2:]) != 3:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 3. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return max_pooling_nd(x, ksize, stride, pad, cover_all, return_indices)
