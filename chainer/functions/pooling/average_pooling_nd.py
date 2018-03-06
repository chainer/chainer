import functools
import operator

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.pooling import average_pooling_nd_kernel
from chainer.functions.pooling import pooling_nd
from chainer import utils
from chainer.utils import conv_nd


class AveragePoolingND(pooling_nd._PoolingND):

    """Average pooling over a set of N-dimensional planes.

    .. warning::

        This feature is experimental. The interface can change in the future.

    """

    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=False):
        utils.experimental('chainer.functions.pooling.AveragePoolingND')

        # TODO(takagi) Support cover_all mode.
        if cover_all is True:
            raise ValueError('`cover_all` mode is not supported yet.')

        super(AveragePoolingND, self).__init__(
            ndim, ksize, stride=stride, pad=pad, cover_all=cover_all)

    def forward_cpu(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        col = conv_nd.im2col_nd_cpu(
            x[0], self.ksize, self.stride, self.pad, cover_all=self.cover_all)

        # mean along (_, _, k_1, k_2, ..., k_N, _, ..., _)
        y_axis = tuple(six.moves.range(2, 2 + len(self.ksize)))
        y = col.mean(axis=y_axis)
        return y,

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('>=auto') and 2 <= self.ndim <= 3:
            # With cuDNN v3 or greater, use cuDNN implementation for inputs
            # with spatial dimensions of two or more.
            self.retain_inputs((0,))
            return super(AveragePoolingND, self).forward_gpu(x)

        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = tuple(conv_nd.get_conv_outsize(d, k, s, p,
                                            cover_all=self.cover_all)
                   for (d, k, s, p) in six.moves.zip(
                       dims, self.ksize, self.stride, self.pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x[0].dtype)
        coeff = 1. / functools.reduce(operator.mul, self.ksize)

        in_params, out_params, operation, name = \
            average_pooling_nd_kernel.AveragePoolingNDKernelForward.generate(
                self.ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x[0].reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad + (coeff, y)))

        return y,

    def backward(self, indexes, gy):
        return AveragePoolingNDGrad(self).apply(gy)

    def create_pool_desc(self):
        return cuda.cudnn.create_pooling_descriptor(
            self.ksize, self.stride, self.pad,
            cuda.cuda.cudnn.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)


class AveragePoolingNDGrad(function_node.FunctionNode):

    def __init__(self, apoolnd):
        self.ndim = apoolnd.ndim
        self.ksize = apoolnd.ksize
        self.stride = apoolnd.stride
        self.pad = apoolnd.pad
        self.cover_all = apoolnd.cover_all
        self._used_cudnn = apoolnd._used_cudnn
        if not self._used_cudnn:
            self._in_shape = apoolnd._in_shape
            self._in_dtype = apoolnd._in_dtype
        self.apoolnd = apoolnd

    def forward_cpu(self, gy):
        dims = self._in_shape[2:]
        outs = gy[0].shape[2:]
        colon = slice(None, None, None)
        gy_index = (colon, colon) + (None,) * len(dims)
        gcol_reps = (1, 1) + self.ksize + (1,) * len(outs)
        gcol = numpy.tile(gy[0][gy_index], gcol_reps)
        gx = conv_nd.col2im_nd_cpu(gcol, self.stride, self.pad, dims)
        gx /= functools.reduce(operator.mul, self.ksize)
        return gx,

    def forward_gpu(self, gy):
        if self._used_cudnn:
            x, = self.apoolnd.get_retained_inputs()
            return self.apoolnd.backward_gpu((x.data,), gy)

        n, c = self._in_shape[:2]
        dims = self._in_shape[2:]
        ys = gy[0].shape[2:]
        gx = cuda.cupy.empty(self._in_shape, self._in_dtype)
        coeff = 1. / functools.reduce(operator.mul, self.ksize)

        in_params, out_params, operation, name = \
            average_pooling_nd_kernel.AveragePoolingNDKernelBackward.generate(
                self.ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            gy[0].reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad + (coeff, gx)))

        return gx,

    def backward(self, indexes, grad_outputs):
        return AveragePoolingND(
            self.ndim, self.ksize, self.stride, self.pad,
            cover_all=False).apply(grad_outputs)


def average_pooling_nd(x, ksize, stride=None, pad=0):
    """N-dimensionally spatial average pooling function.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This function provides a N-dimensionally generalized version of
    :func:`~functions.average_pooling_2d`. This acts similarly to
    :class:`~functions.ConvolutionND`, but it computes the average of input
    spatial patch for each channel without any parameter instead of computing
    the inner products.

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

    Returns:
        ~chainer.Variable: Output variable.

    .. note::

       This function currently does not support ``cover_all`` mode as
       :func:`max_pooling_nd`. Average pooling runs in non-cover-all mode.

    """
    ndim = len(x.shape[2:])
    return AveragePoolingND(ndim, ksize, stride=stride, pad=pad).apply((x,))[0]
