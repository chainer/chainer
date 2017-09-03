from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check

import numpy
import six


class Upsampling2D(pooling_2d.Pooling2D):

    """Upsampling over a set of 2d planes w/ indices used for max pooling."""

    def __init__(self, indexes, ksize, stride=None, pad=0, outsize=None,
                 cover_all=True):
        super(Upsampling2D, self).__init__(ksize, stride, pad, cover_all)
        self.indexes = indexes
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
            x_type.shape == self.indexes.shape,
        )

        if self.outh is not None:
            expected_h = conv.get_conv_outsize(
                self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(
                self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)

    def forward(self, x):
        self.retain_inputs(())
        xp = cuda.get_array_module(*x)
        if isinstance(x[0], numpy.ndarray):
            im2col, col2im = conv.im2col_cpu, conv.col2im_cpu
        else:
            im2col, col2im = conv.im2col_gpu, conv.col2im_gpu
        self._in_dtype = x[0].dtype

        n, c, h, w = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)

        up_y = xp.zeros((n, c, self.outh, self.outw), dtype=self._in_dtype)
        up_y = im2col(
            up_y, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all).transpose(0, 1, 4, 5, 2, 3)
        colh, colw = up_y.shape[2:4]
        up_y = up_y.reshape(-1, self.kh * self.kw)
        indexes = self.indexes.ravel()
        up_y[xp.arange(len(indexes)), indexes] = x[0].ravel()
        up_y = up_y.reshape(n, c, colh, colw, self.kh, self.kw)
        up_y = col2im(
            up_y.transpose(0, 1, 4, 5, 2, 3), self.sy, self.sx, self.ph,
            self.pw, self.outh, self.outw)
        return up_y,

    def backward(self, x, gy):
        im2col = conv.im2col_cpu \
            if isinstance(gy[0], numpy.ndarray) else conv.im2col_gpu
        xp = cuda.get_array_module(*x)
        gcol = im2col(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)

        n, c, kh, kw, out_h, out_w = gcol.shape
        gcol = gcol.transpose(0, 1, 4, 5, 2, 3).reshape(-1, kh * kw)
        indexes = self.indexes.ravel()
        gx = gcol[xp.arange(len(indexes)), indexes]
        return gx.reshape(n, c, out_h, out_w),


def upsampling_2d(
        x, indexes, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Upsampling using pooling indices.

    This function produces an upsampled image using pooling indices.

    .. admonition:: Example

        It should be noted that you need to turn off
        ``chainer.config.use_cudnn`` flag when you perform
        :meth:`~chainer.functions.max_pooling_2d` function which will make a
        pooling indicies for this :meth:`~chainer.functions.upsampling_2d`.
        It is because :attr:`~chainer.functions.MaxPooling2D.indexes` is never
        created and stored in the :attr:`~chainer.functions.MaxPooling2D`
        object when cuDNN is used for it.

        >>> x = np.arange(1, 37).reshape(1, 1, 6, 6).astype('f')
        >>> x = chainer.Variable(x)
        >>> x.data
        array([[[[  1.,   2.,   3.,   4.,   5.,   6.],
                 [  7.,   8.,   9.,  10.,  11.,  12.],
                 [ 13.,  14.,  15.,  16.,  17.,  18.],
                 [ 19.,  20.,  21.,  22.,  23.,  24.],
                 [ 25.,  26.,  27.,  28.,  29.,  30.],
                 [ 31.,  32.,  33.,  34.,  35.,  36.]]]], dtype=float32)

        This is the original ``x`` before max pooling.

        >>> p = F.MaxPooling2D(2, 2)
        >>> with chainer.using_config('use_cudnn', 'never'):
        ...     pooled_x = p(x)
        >>> pooled_x.data
        array([[[[  8.,  10.,  12.],
                 [ 20.,  22.,  24.],
                 [ 32.,  34.,  36.]]]], dtype=float32)

        This is the output of the max pooling operation.
        :meth:`~chainer.functions.upsampling_2d` needs
        :attr:`~chainer.functions.MaxPooling2D.indexes` array stored in the max
        pooling object ``p``.

        >>> upsampled_x = F.upsampling_2d(
        ...     pooled_x, p.indexes, p.kh, p.sy, p.ph, x.shape[2:])
        >>> upsampled_x.shape
        (1, 1, 6, 6)
        >>> upsampled_x.data
        array([[[[  0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   8.,   0.,  10.,   0.,  12.],
                 [  0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,  20.,   0.,  22.,   0.,  24.],
                 [  0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,  32.,   0.,  34.,   0.,  36.]]]], dtype=float32)

    Args:
        x (~chainer.Variable): Input variable.
        indexes (~numpy.ndarray or ~cupy.ndarray): Index array that was used
            to calculate x with MaxPooling2D.
        ksize (int or (int, int)): ksize attribute of MaxPooling2D object that
            is used to calculate x
        stride (int or (int, int)): stride attribute of MaxPooling2D object
            that is used to calculate x
        pad (int or (int, int)): pad attribute of MaxPooling2D object that is
            used to calculate x
        outsize ((int, int)): Expected output size (height, width).
        cover_all (bool): Whether cover_all is used in the MaxPooling2D object
            or not.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Upsampling2D(indexes, ksize, stride, pad, outsize, cover_all)(x)
