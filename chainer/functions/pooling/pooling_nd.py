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

    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=True,
                 return_indices=False):
        if stride is None:
            stride = ksize

        if ndim <= 0:
            raise ValueError(
                'pooling operation requires at least one spatial dimension.')

        self.ndim = ndim
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)

        self.cover_all = cover_all
        self.return_indices = return_indices

        self._used_cudnn = False

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 2 + self.ndim,
            in_types[0].size > 0,
        )

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        self._used_cudnn = True

        # Implementation using cuDNN.
        x = x[0]
        n, c = x.shape[:2]
        dims = x.shape[2:]
        ys = tuple(conv.get_conv_outsize(d, k, s, p, self.cover_all)
                   for d, k, s, p in six.moves.zip(
                       dims, self.ksize, self.stride, self.pad))
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)

        cudnn.pooling_forward(
            x, y, self.ksize, self.stride, self.pad, self._get_pool_mode())
        self.retain_outputs((0,))
        return y,

    def backward_gpu(self, x, gy):
        # Implementation using cudnn
        x = x[0]
        y = self.get_retained_outputs()[0].array
        gx = cudnn.pooling_backward(
            x, y, gy[0],
            self.ksize, self.stride, self.pad, self._get_pool_mode())
        return gx,

    def _get_pool_mode(self):
        raise NotImplementedError()
