import numpy
import six

from chainer.backends import cuda


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of convolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_deconv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of convolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the convolution operation.

    """
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def get_deconv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of deconvolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_conv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of deconvolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the deconvolution operation.

    """
    dk = (k - 1) * d + 1
    if cover_all:
        return s * (size - 1) + dk - s + 1 - 2 * p
    else:
        return s * (size - 1) + dk - 2 * p


def im2col_cpu(
        img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return col


def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False, dy=1, dx=1,
               out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    cuda.elementwise(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)
    return col


def im2col(img, kh, kw, sy, sx, ph, pw, cover_all=False, dy=1, dx=1,
           out_h=None, out_w=None):
    fn = im2col_gpu if isinstance(img, cuda.ndarray) else im2col_cpu
    return fn(img, kh, kw, sy, sx, ph, pw, cover_all=cover_all, dy=dy, dx=dx,
              out_h=out_h, out_w=out_w)


def col2im_cpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]


def col2im_gpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)
    cuda.elementwise(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


def col2im(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    fn = col2im_gpu if isinstance(col, cuda.ndarray) else col2im_cpu
    return fn(col, sy, sx, ph, pw, h, w, dy, dx)
