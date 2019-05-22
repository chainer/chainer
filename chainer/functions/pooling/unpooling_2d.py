import numpy
import numpy.lib.stride_tricks
try:
    import cupy.lib.stride_tricks  # NOQA
except Exception:
    pass

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check


class Unpooling2D(pooling_2d.Pooling2D):

    """Unpooling over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0,
                 outsize=None, cover_all=True):
        super(Unpooling2D, self).__init__(ksize, stride, pad, cover_all)
        self.outh, self.outw = (None, None) if outsize is None else outsize
        self._use_int_scale_forward = False

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
        )

        if self.outh is not None:
            expected_h = conv.get_conv_outsize(
                self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(
                self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)

    def _integer_scale_forward(self, x):
        xp = backend.get_array_module(x)
        b, c, h, w = x.shape
        bs, cs, hs, ws = x.strides
        if self.ph > 0 or self.pw > 0:
            x = x[:, :, self.ph // 2:-self.ph // 2, self.pw // 2:-self.pw // 2]
        y = xp.lib.stride_tricks.as_strided(
            x,
            (b, c, h - self.ph, self.kh, w - self.pw, self.kw),
            (bs, cs, hs, 0, ws, 0))
        y = y.reshape((b, c, self.kh * (h - self.ph), self.kw * (w - self.pw)))
        return y,

    def forward(self, x):
        h, w = x[0].shape[2:]
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)
        if (self.outh % (h - self.ph) == 0 and
            self.outw % (w - self.pw) == 0 and
            self.outh // (h - self.ph) == self.kh and
            self.outw // (w - self.pw) == self.kw and
            self.ph % 2 == 0 and self.pw % 2 == 0 and
                self.sx == self.kh and self.sy == self.kw):
            self._use_int_scale_forward = True
            return self._integer_scale_forward(x[0])
        xp = backend.get_array_module(*x)
        col = xp.tile(x[0][:, :, None, None],
                      (1, 1, self.kh, self.kw, 1, 1))
        if xp is numpy:
            y = conv.col2im_cpu(col, self.sy, self.sx, self.ph, self.pw,
                                self.outh, self.outw)
        else:
            y = conv.col2im_gpu(col, self.sy, self.sx, self.ph, self.pw,
                                self.outh, self.outw)
        return y,

    def backward(self, indexes, grad_outputs):
        return Unpooling2DGrad(self).apply(grad_outputs)


class Unpooling2DGrad(function_node.FunctionNode):

    def __init__(self, unpooling2d):
        self.kh = unpooling2d.kh
        self.kw = unpooling2d.kw
        self.sy = unpooling2d.sy
        self.sx = unpooling2d.sx
        self.ph = unpooling2d.ph
        self.pw = unpooling2d.pw
        self.outh = unpooling2d.outh
        self.outw = unpooling2d.outw
        self.cover_all = unpooling2d.cover_all
        self._use_int_scale_forward = unpooling2d._use_int_scale_forward

    def _integer_scale_forward(self, gy):
        xp = backend.get_array_module(gy)
        b, c, h, w = gy.shape
        gx = gy.reshape((b, c, h // self.kh, self.kh, w // self.kw, self.kw))
        gx = xp.rollaxis(gx, 3, 5).sum((4, 5))
        if self.ph > 0 or self.pw > 0:
            tmp = xp.zeros((b, c, h // 2 + self.ph, w //
                            2 + self.pw), dtype=gx.dtype)
            tmp[:, :, self.ph // 2:-self.ph // 2,
                self.pw // 2:-self.pw // 2] = gx
            gx = tmp
        return gx,

    def forward(self, gy):
        if self._use_int_scale_forward:
            return self._integer_scale_forward(gy[0])
        if isinstance(gy[0], cuda.ndarray):
            gcol = conv.im2col_gpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        else:
            gcol = conv.im2col_cpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        gx = gcol.sum(axis=(2, 3))
        return gx,

    def backward(self, indexes, ggx):
        return Unpooling2D(
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            (self.outh, self.outw), self.cover_all).apply(ggx)


def unpooling_2d(x, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Inverse operation of pooling for 2d array.

    This function acts similarly to
    :class:`~functions.connection.deconvolution_2d.Deconvolution2DFunction`,
    but it spreads input 2d array's value without any parameter instead of
    computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int, pair of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is
            specified, then it uses same stride as the pooling window size.
        pad (int or pair of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        outsize (None or pair of ints): Expected output size (height, width)
            of array after the operation.  If ``None``, the size
            (height or width) is estimated from the size of input array
            in first batch with
            :func:`~chainer.utils.conv.get_deconv_outsize`.
            If outsize is not ``None``, the result of outsize applied to
            :func:`~chainer.utils.conv.get_conv_outsize` must be equal to
            the shape of the 2d array in the input batch ``x``.
        cover_all (bool): If ``True``, the output size may be smaller than
            the size if ``cover_all`` is ``False``. This flag serves to
            align behavior to the pooling functions which can cover all
            input locations, see :func:`~chainer.functions.max_pooling_2d`
            and :func:`~chainer.functions.convolution_2d`.


    Returns:
        ~chainer.Variable: Output variable.

    """
    return Unpooling2D(ksize, stride, pad, outsize, cover_all).apply((x,))[0]
