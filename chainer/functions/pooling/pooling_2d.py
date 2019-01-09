from chainer.backends import cuda
from chainer import function_node
from chainer.utils import collections_abc
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn


def _pair(x):
    if isinstance(x, collections_abc.Iterable):
        return x
    return x, x


class Pooling2D(function_node.FunctionNode):

    """Base class of pooling function over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0, cover_all=True,
                 return_indices=False):
        if stride is None:
            stride = ksize

        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

        self.cover_all = cover_all
        self.return_indices = return_indices

        self._used_cudnn = False

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        self._used_cudnn = True

        # Implementation using cudnn
        x = x[0]
        n, c, h, w = x.shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        assert y_w > 0, 'Width in the output should be positive.'
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x.dtype)

        cudnn.pooling_forward(
            x, y,
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            self._get_pool_mode())
        self.retain_outputs((0,))
        return y,

    def backward_gpu(self, x, gy):
        # Implementation using cudnn
        x = x[0]
        y = self.get_retained_outputs()[0].array
        gx = cudnn.pooling_backward(
            x, y, gy[0],
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            self._get_pool_mode())
        return gx,

    def _get_pool_mode(self):
        raise NotImplementedError()
