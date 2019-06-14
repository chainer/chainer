import numpy

from chainer import function_node
from chainer.utils.conv import col2im_cpu
from chainer.utils.conv import col2im_gpu
from chainer.utils.conv import im2col_cpu
from chainer.utils.conv import im2col_gpu
from chainer.utils import type_check


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


def _col2im(x, *args, **kwargs):
    if isinstance(x, numpy.ndarray):
        return col2im_cpu(x, *args, **kwargs)
    return col2im_gpu(x, *args, **kwargs)


def _im2col(x, *args, **kwargs):
    if isinstance(x, numpy.ndarray):
        return im2col_cpu(x, *args, **kwargs)
    return im2col_gpu(x, *args, **kwargs)


class Im2Col(function_node.FunctionNode):

    """Im2Col function."""

    def __init__(self, ksize, stride, pad, cover_all, dilate):
        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.dy, self.dx = _pair(dilate)
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4
        )

    def forward(self, inputs):
        x, = inputs
        y = _im2col(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        n, c, kh, kw, out_h, out_w = y.shape
        return y.reshape(n, c * kh * kw, out_h, out_w),

    def backward(self, indexes, grad_outputs):
        return Im2ColGrad((self.kh, self.kw), (self.sy, self.sx),
                          (self.ph, self.pw), self.cover_all,
                          (self.dy, self.dx), self.inputs[0].shape) \
            .apply(grad_outputs)


class Im2ColGrad(function_node.FunctionNode):

    """Im2Col gradient function."""

    def __init__(self, ksize, stride, pad, cover_all, dilate, in_shape):
        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.dy, self.dx = _pair(dilate)
        self.cover_all = cover_all
        self.in_shape = in_shape

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('gy',))

        gy_type = in_types[0]
        type_check.expect(
            gy_type.dtype.kind == 'f',
            gy_type.ndim == 4
        )

    def forward(self, inputs):
        _, c, h, w = self.in_shape
        gy, = inputs
        n, _, out_h, out_w = gy.shape
        gy = gy.reshape(n, c, self.kh, self.kw, out_h, out_w)
        gx = _col2im(
            gy, self.sy, self.sx, self.ph, self.pw, h, w, self.dy, self.dx)
        return gx,

    def backward(self, indexes, grad_outputs):
        return Im2Col(
            (self.kh, self.kw), (self.sy, self.sx),
            (self.ph, self.pw), self.cover_all,
            (self.dy, self.dx)).apply(grad_outputs)


def im2col(x, ksize, stride=1, pad=0, cover_all=False, dilate=1):
    """Extract patches from an image based on the filter.

    This function rearranges patches of an image and puts them in the channel
    dimension of the output.

    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``-pad`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Notation: here is a notation.

    - :math:`n` is the batch size.
    - :math:`c` is the number of the input channels.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.
    - :math:`s_Y` and :math:`s_X` are the strides of the filter.
    - :math:`p_H` and :math:`p_W` are the spatial padding sizes.
    - :math:`d_Y` and :math:`d_X` are the dilation factors of filter \
        application.

    The output size :math:`(h_O, w_O)` is determined by the following
    equations when ``cover_all = False``:

    .. math::

       h_O &= (h + 2p_H - k_H - (k_H - 1) * (d_Y - 1)) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W - (k_W - 1) * (d_X - 1)) / s_X + 1.

    When ``cover_all = True``, the output size is determined by
    the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H - (k_H - 1) * (d_Y - 1) + s_Y - 1) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W - (k_W - 1) * (d_X - 1) + s_X - 1) / s_X + 1.


    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable of shape :math:`(n, c, h, w)`.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (bool): If ``True``, all spatial locations are rearranged
            into some output pixels. It may make the output size larger.
        dilate (int or pair of ints): Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.

    Returns:
        ~chainer.Variable:
        Output variable whose shape is
        :math:`(n, c \\cdot k_H \\cdot k_W, h_O, w_O)`

    """
    return Im2Col(ksize, stride, pad, cover_all, dilate).apply((x,))[0]
