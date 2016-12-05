import numpy
from six import moves

from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class DepthwiseConvolution2D(function.Function):

    def __init__(self, stride=1, pad=0, cover_all=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw = W.shape[2:]

        xp = cuda.get_array_module(*x)
        if xp is numpy:
            self.col = conv.im2col_cpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        else:
            self.col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)

        y = xp.tensordot(
            self.col, W, ((2, 3), (2, 3))).astype(x.dtype, copy=False)
        y = xp.concatenate(y, axis=1)  # TODO
        if b is not None:
            y += b
        return xp.rollaxis(y, 3, 1),

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        xp = cuda.get_array_module(*x)
        gy = xp.split(gy, axis=1)  # TODO
        gW = xp.tensordot(
            gy, self.col, ((0, 2, 3), (0, 1, 4, 5))).astype(W.dtype, copy=False)
        gcol = xp.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)  # TODO
        gcol = xp.rollaxis(gcol, 3)

        if xp is numpy:
            gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        else:
            gx = conv.col2im_gpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb


def depthwise_convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True,
                   cover_all=False, deterministic=False):
    """Two-dimensional convolution function.

    This is an implementation of two-dimensional convolution in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_I, k_H, k_W)`.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.


    Returns:
        ~chainer.Variable: Output variable.

    The two-dimensional convolution function is defined as follows.
    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(k_H, k_W)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``-pad`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    .. seealso:: :class:`~chainer.links.DepthwiseConvolution2D`

    """
    func = DepthwiseConvolution2D(stride, pad, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
