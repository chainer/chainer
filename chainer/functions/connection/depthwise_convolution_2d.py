import numpy

from chainer.backends import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


def _matmul(a, b, xp):
    if xp is numpy:
        # numpy 1.9 does not support matmul.
        # So we use numpy.einsum instead of numpy.matmul.
        return xp.einsum('ijk,ikl->ijl', a, b)
    else:
        return xp.matmul(a, b)


class DepthwiseConvolution2D(function.Function):

    def __init__(self, stride=1, pad=0, direct=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.direct = direct

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

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0] * w_type.shape[1],
            )

    def forward(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw = W.shape[2:]

        xp = cuda.get_array_module(*x)
        if xp is numpy:
            self.col = conv.im2col_cpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)
        elif self.direct:
            return self.forward_gpu_direct(inputs)
        else:
            self.col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)

        B, C, KY, KX, IY, IX = self.col.shape
        D = W.shape[0]  # (D, C, KY, KX)
        c_ = self.col.transpose(1, 0, 4, 5, 2, 3) \
            .reshape((C, B * IY * IX, KY * KX))
        w_ = W.transpose(1, 2, 3, 0).reshape((C, KY * KX, D))

        # (C, B*IY*IX, KY*KX), (C, KY*KX, D)-> (C, B*IY*IX, D)
        y = _matmul(c_, w_, xp).astype(x.dtype, copy=False)

        # (C, B*IY*IX, D) -> (B, C*D, IY, IX)
        y = y.reshape((C, B, IY * IX, D)).transpose(1, 0, 3, 2) \
            .reshape((B, C * D, IY, IX))

        if b is not None:
            y += b[None, :, None, None]
        return y,

    def forward_gpu_direct(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        c_m, c_i, kh, kw = W.shape

        n, c, h, w = x.shape
        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph)
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw)
        y = cuda.cupy.zeros((n, c_m * c_i, out_h, out_w), dtype=x.dtype)
        cuda.elementwise(
            'raw T W, raw U img,'
            'int32 n, int32 c, int32 h, int32 w,'
            'int32 m, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'V out',
            '''
                const int n_i = i / (c*m*out_h*out_w);
                const int out_c = i / (out_h*out_w) % (m*c);
                const int in_c = out_c / m;
                const int m_i = out_c % m;
                const int out_y = i / out_w % out_h;
                const int out_x = i % out_w;
                int widx = (m_i*c + in_c)*kw*kh;
                for(int ky=0; ky<kh; ky++) {
                  const int in_y = out_y * sy - ph + ky;
                  for(int kx=0; kx<kw; kx++) {
                    const int in_x = out_x * sx - pw + kx;
                    if (0<=in_x && in_x<w && 0<=in_y && in_y<h) {
                      const int idx = ((n_i*c + in_c)*h + in_y)*w + in_x;
                      out += W[widx] * img[idx];
                    }
                    widx++;
                  }
                }
            ''',
            'direct_depthwise_conv')(W.reduced_view(),
                                     x.reduced_view(),
                                     n, c_i, h, w, c_m, out_h, out_w,
                                     kh, kw, self.sy, self.sx,
                                     self.ph, self.pw, y)
        if b is not None:
            y += b[None, :, None, None]
        return y,

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        xp = cuda.get_array_module(*x)
        if xp == cuda.cupy and self.direct:
            return self.backward_gpu_direct(inputs, grad_outputs)

        B, C, KY, KX, IY, IX = self.col.shape
        D = W.shape[0]

        # (B, C*D, IY, IX) -> (C, D, B*IY*IX, D)
        gy_ = gy.reshape((B, C, D, IY * IX)).transpose(1, 2, 0, 3) \
            .reshape((C, D, B * IY * IX))
        c_ = self.col.transpose(1, 0, 4, 5, 2, 3) \
            .reshape((C, B * IY * IX, KY * KX))
        # (C, D, B*IY*IX), (C, B*IY*IX, KY*KX) -> (C, D, KY*KX)
        gW_ = _matmul(gy_, c_, xp)
        gW = gW_.reshape((C, D, KY, KX)).transpose(1, 0, 2, 3)
        gW = gW.astype(W.dtype, copy=False)

        w_ = W.transpose(1, 2, 3, 0).reshape((C, KY * KX, D))
        # (C, KY*KX, D), (C, D, B*IY*IX) -> (C, KY*KX, B*IY*IX)
        gcol = _matmul(w_, gy_, xp).reshape((C, KY, KX, B, IY, IX))
        gcol = gcol.astype(x.dtype, copy=False)
        gcol = xp.rollaxis(gcol, 3)

        if xp is numpy:
            gx = conv.col2im_cpu(gcol, self.sy, self.sx,
                                 self.ph, self.pw, h, w)
        else:
            gx = conv.col2im_gpu(gcol, self.sy, self.sx,
                                 self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gy = xp.rollaxis(gy, 1, 4)
            gb = gy.sum(axis=(0, 1, 2))
            return gx, gW, gb

    def backward_gpu_direct(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        n, c, h, w = x.shape
        m, c, kh, kw = W.shape
        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph)
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw)
        gW = cuda.cupy.zeros((m, c, kh, kw), dtype=W.dtype)
        xp = cuda.get_array_module(*x)

        cuda.elementwise(
            'raw T gout, raw U img,'
            'int32 n, int32 c, int32 h, int32 w,'
            'int32 m, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'V gW',
            '''
                const int m_i = i / (c*kh*kw);
                const int in_c = i / (kh*kw) % c;
                const int go_c = in_c * m + m_i;
                const int ky = i / kw % kh;
                const int kx = i % kw;
                const int go_pane_size = out_h * out_w;
                for (int n_i=0; n_i<n; n_i++) {
                  for (int go_pos=0; go_pos<go_pane_size; go_pos++) {
                    const int go_x = go_pos % out_w;
                    const int go_y = go_pos / out_w;
                    const int in_x = go_x * sx - pw + kx;
                    const int in_y = go_y * sy - ph + ky;
                    if (0<=in_x && in_x<w && 0<=in_y && in_y<h) {
                      const int idx = ((n_i*c + in_c)*h + in_y)*w + in_x;
                      const int gidx = (n_i*c*m + go_c)*go_pane_size + go_pos;
                      gW += gout[gidx] * img[idx];
                    }
                  }
                }
            ''',
            'direct_depthwise_gW')(gy.reduced_view(),
                                   x.reduced_view(),
                                   n, c, h, w, m, out_h, out_w,
                                   kh, kw, self.sy, self.sx,
                                   self.ph, self.pw, gW)
        gx = cuda.cupy.zeros((n, c, h, w), dtype=x.dtype)
        cuda.elementwise(
            'raw T gout, raw U W,'
            'int32 n, int32 c, int32 h, int32 w,'
            'int32 m, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'V gx',
            '''
                const int n_i = i / (c*h*w);
                const int in_c = i / (h*w) % c;
                const int in_y = i / w % h;
                const int in_x = i % w;
                for(int m_i=0; m_i<m; m_i++) {
                  const int go_c = in_c*m+m_i;
                  int widx = (m_i*c + in_c) * kh * kw;
                  for(int ky=0; ky<kh; ky++) {
                    for(int kx=0; kx<kw; kx++) {
                      int go_y = in_y + ph - ky;
                      int go_x = in_x + pw - kx;
                      if((go_y % sy == 0) && (go_x % sx == 0)){
                        go_y = go_y / sy;
                        go_x = go_x / sx;
                        if(0<=go_x && go_x<out_w && 0<=go_y && go_y<out_h) {
                          const int gidx =
                            ((n_i*m*c + go_c)*out_h + go_y)*out_w + go_x;
                          gx += gout[gidx] * W[widx];
                        }
                      }
                      widx++;
                    }
                  }
                }
            ''',
            'direct_depthwise_gx')(gy.reduced_view(),
                                   W.reduced_view(),
                                   n, c, h, w, m, out_h, out_w,
                                   kh, kw, self.sy, self.sx,
                                   self.ph, self.pw, gx)

        if b is None:
            return gx, gW
        else:
            gy = xp.rollaxis(gy, 1, 4)
            gb = gy.sum(axis=(0, 1, 2))
            return gx, gW, gb


def depthwise_convolution_2d(x, W, b=None, stride=1, pad=0, direct=False):
    """Two-dimensional depthwise convolution function.

    This is an implementation of two-dimensional depthwise convolution.
    It takes two or three variables: the input image ``x``, the filter weight
    ``W``, and optionally, the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input.
    - :math:`c_M` is the channel multiplier.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_M, c_I, k_H, k_W)`.
        b (~chainer.Variable):
            Bias variable of length :math:`c_M * c_I` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        direct (bool): If ``True`` and gpu enabled, this function use direct
            implementation of convolution instead of im2col.
            This is efficient when the channel size is large.

    Returns:
        ~chainer.Variable:
            Output variable. Its shape is :math:`(n, c_I * c_M, h_O, w_O)`.

    Like ``Convolution2D``, ``DepthwiseConvolution2D`` function computes
    correlations between filters and patches of size :math:`(k_H, k_W)` in
    ``x``.
    But unlike ``Convolution2D``, ``DepthwiseConvolution2D`` does not add up
    input channels of filters but concatenates them.
    For that reason, the shape of outputs of depthwise convolution are
    :math:`(n, c_I * c_M, h_O, w_O)`, :math:`c_M` is called channel_multiplier.

    :math:`(h_O, w_O)` is determined by the equivalent equation of
    ``Convolution2D``.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    See: `L. Sifre. Rigid-motion scattering for image classification\
          <http://www.di.ens.fr/data/publications/papers/phd_sifre.pdf>`_

    .. seealso:: :class:`~chainer.links.DepthwiseConvolution2D`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (2, 3, 4, 7))
        >>> W = np.random.uniform(0, 1, (2, 3, 3, 3))
        >>> b = np.random.uniform(0, 1, (6,))
        >>> y = F.depthwise_convolution_2d(x, W, b)
        >>> y.shape
        (2, 6, 2, 5)

    """
    func = DepthwiseConvolution2D(stride, pad, direct)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
