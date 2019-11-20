import six

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn


class _PoolingND(function_node.FunctionNode):

    """Base class of pooling function over a set of N-dimensional planes."""

    _cudnn_pool = None

    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=True,
                 return_indices=False):
        if stride is None:
            stride = ksize

        if ndim <= 0:
            raise ValueError(
                'pooling operation requires at least one spatial dimension.')

        super(_PoolingND, self).__init__()

        self.ndim = ndim
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)

        self.cover_all = cover_all
        self.return_indices = return_indices

    @property
    def is_cudnn_used(self):
        return self._cudnn_pool is not None

    def get_cudnn_pool_mode(self):
        raise NotImplementedError()

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 2 + self.ndim,
            in_types[0].size > 0,
        )

    def forward_cudnn(self, x):
        self._cudnn_pool = _CudnnPoolingNDImpl(self)
        return self._cudnn_pool.forward(x)

    def backward_cudnn(self, gy):
        assert self._cudnn_pool is not None
        return self._cudnn_pool.backward(gy)


class _CudnnPoolingNDImpl(object):

    """cuDNN pooling implementation"""

    def __init__(self, func):
        assert isinstance(func, _PoolingND)
        self.func = func

    def forward(self, x):
        func = self.func
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        cover_all = func.cover_all
        pool_mode = func.get_cudnn_pool_mode()

        x = x[0]
        n, c = x.shape[:2]
        dims = x.shape[2:]
        ys = tuple(conv.get_conv_outsize(d, k, s, p, cover_all)
                   for d, k, s, p in six.moves.zip(dims, ksize, stride, pad))
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)

        cudnn.pooling_forward(x, y, ksize, stride, pad, pool_mode)
        func.retain_inputs((0,))
        func.retain_outputs((0,))
        return y,

    def backward(self, gy):
        func = self.func
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        pool_mode = func.get_cudnn_pool_mode()

        x = func.get_retained_inputs()[0].array
        y = func.get_retained_outputs()[0].array
        gx = cudnn.pooling_backward(x, y, gy[0], ksize, stride, pad, pool_mode)
        return gx,
