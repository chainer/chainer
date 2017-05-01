import numpy

from chainer import cuda

from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.array.concat import concat
from chainer.functions.array.pad import pad as pad_func
from chainer.functions.array.spatial_transformer_sampler import\
    spatial_transformer_sampler
from chainer.functions.math.matmul import matmul


def deformable_convolution_2d_sampler(x, offset, W, b=None, stride=1, pad=0,
                                      use_cudnn=True):
    """Two-dimensional deformable convolution function using computed offset.

    This is implementation of two-dimensional deformable convolution used in
    `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    It takes four variables: the input image ``x``, the offset image
    ``offset``, the filter weight ``W``, and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output,
      respectively.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.
    - :math:`s_Y` and :math:`s_X` are the stride of the filter.
    - :math:`p_H` and :math:`p_W` are the spatial padding size.

    The output size :math:`(h_O, w_O)` is determined by the following
    equations:

    .. math::

       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W) / s_X + 1.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
        offset (~chainer.Variable): Offset variable of shape
            :math:`(n, 2 \\cdot k_H \\cdot k_W, h_O, w_O)`. The first
            :math:`k_H \\cdot k_W` index of the second axis corresponds to
            the offsets in the horizontal direction. The last
            :math:`k_H \\cdot k_W` index of the second axis corresponds to
            the offsets in the vertical direction.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_O, c_I, k_H, k_W)`.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If ``True``, then this function uses cuDNN for
            :obj:`spatial_transformer_sampler` if available.
            Note that, cuDNN supports :obj:`spatial_transformer_sampler`
            from version 5.0.0.

    Returns:
        ~chainer.Variable: Output variable.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    .. seealso:: :class:`DeformableConvolution2DSampler`

    """
    sy, sx = _pair(stride)
    ph, pw = _pair(pad)
    out_c, _, kh, kw = W.shape
    n, c, h, w = x.shape
    _, _, out_h, out_w = offset.shape

    grid = _offset2grid(offset, kh, kw, sy, sx, ph, pw, h, w)
    grid = grid.reshape(n, 2, kh * kw, out_h * out_w)
    x_pad = pad_func(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), 'constant')
    x_st = spatial_transformer_sampler(x_pad, grid, use_cudnn)

    x_st = x_st.transpose(0, 3, 1, 2).reshape(n * out_h * out_w, c * kh * kw)
    W = W.transpose(1, 2, 3, 0).reshape(c * kh * kw, out_c)
    y = matmul(x_st, W)
    y = y.reshape(n, out_h, out_w, out_c).transpose(0, 3, 1, 2)

    if b is not None:
        b = broadcast_to(b[None, :, None, None], y.shape)
        y += b
    return y


def _offset2grid(offset, kh, kw, sy, sx, ph, pw, h, w):
    n, khkw2, out_h, out_w = offset.shape
    khkw = khkw2 / 2
    xp = cuda.get_array_module(offset)

    ys, xs = xp.meshgrid(
        xp.arange(0, sy * out_h, sy, dtype=numpy.float32),
        xp.arange(0, sx * out_w, sx, dtype=numpy.float32), indexing='ij',
        copy=False
    )
    filter_offset_x = xp.tile(xp.arange(kw), kh)
    filter_offset_y = xp.repeat(xp.arange(kh), kw)
    x_coord = (offset[:, :khkw] + xs[None, None] +
               filter_offset_x[None, :, None, None])
    y_coord = (offset[:, khkw:] + ys[None, None] +
               filter_offset_y[None, :, None, None])

    # The values of this variable is clipped in range [-1, 1].
    # The coordinate (-1, -1) corresponds to the upper-left
    # corner of the input image.
    x_coord = (x_coord / (w + 2 * pw - 1) - 0.5) * 2
    y_coord = (y_coord / (h + 2 * ph - 1) - 0.5) * 2

    # Shape of `coord` is (n, 2 * kh * kw, out_h, out_w)
    coord = concat([x_coord, y_coord], axis=1)
    return coord


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
