import numpy

from chainer.functions.activation import relu
from chainer.functions.array import concat
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.normalization import batch_normalization


class InceptionBN(link.Chain):

    """Inception module of the new GoogLeNet with BatchNormalization.

    This chain acts like :class:`Inception`, while InceptionBN uses the
    :class:`BatchNormalization` on top of each convolution, the 5x5 convolution
    path is replaced by two consecutive 3x3 convolution applications, and the
    pooling method is configurable.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing \
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

    Args:
        in_channels (int or None): Number of channels of input arrays.
        out1 (int): Output size of the 1x1 convolution path.
        proj3 (int): Projection size of the single 3x3 convolution path.
        out3 (int): Output size of the single 3x3 convolution path.
        proj33 (int): Projection size of the double 3x3 convolutions path.
        out33 (int): Output size of the double 3x3 convolutions path.
        pooltype (str): Pooling type. It must be either ``'max'`` or ``'avg'``.
        proj_pool (int or None): Projection size in the pooling path. If
            ``None``, no projection is done.
        stride (int): Stride parameter of the last convolution of each path.
        conv_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the convolution matrix weights.
            Maybe be ``None`` to use default initialization.
        dtype (numpy.dtype): Type to use in
            ``~batch_normalization.BatchNormalization``.

    .. seealso:: :class:`Inception`

    """

    def __init__(self, in_channels, out1, proj3, out3, proj33, out33,
                 pooltype, proj_pool=None, stride=1, conv_init=None,
                 dtype=numpy.float32):
        super(InceptionBN, self).__init__()
        self.out1 = out1
        self.proj_pool = proj_pool
        self.stride = stride
        self.pooltype = pooltype
        if pooltype != 'max' and pooltype != 'avg':
            raise NotImplementedError()

        with self.init_scope():
            self.proj3 = convolution_2d.Convolution2D(
                in_channels, proj3, 1, nobias=True, initialW=conv_init)
            self.conv3 = convolution_2d.Convolution2D(
                proj3, out3, 3, pad=1, stride=stride, nobias=True,
                initialW=conv_init)
            self.proj33 = convolution_2d.Convolution2D(
                in_channels, proj33, 1, nobias=True, initialW=conv_init)
            self.conv33a = convolution_2d.Convolution2D(
                proj33, out33, 3, pad=1, nobias=True, initialW=conv_init)
            self.conv33b = convolution_2d.Convolution2D(
                out33, out33, 3, pad=1, stride=stride, nobias=True,
                initialW=conv_init)
            self.proj3n = batch_normalization.BatchNormalization(
                proj3, dtype=dtype)
            self.conv3n = batch_normalization.BatchNormalization(
                out3, dtype=dtype)
            self.proj33n = batch_normalization.BatchNormalization(
                proj33, dtype=dtype)
            self.conv33an = batch_normalization.BatchNormalization(
                out33, dtype=dtype)
            self.conv33bn = batch_normalization.BatchNormalization(
                out33, dtype=dtype)

            if out1 > 0:
                assert stride == 1
                assert proj_pool is not None
                self.conv1 = convolution_2d.Convolution2D(
                    in_channels, out1, 1, stride=stride, nobias=True,
                    initialW=conv_init)
                self.conv1n = batch_normalization.BatchNormalization(
                    out1, dtype=dtype)

            if proj_pool is not None:
                self.poolp = convolution_2d.Convolution2D(
                    in_channels, proj_pool, 1, nobias=True, initialW=conv_init)
                self.poolpn = batch_normalization.BatchNormalization(
                    proj_pool, dtype=dtype)

    def __call__(self, x):
        outs = []

        if self.out1 > 0:
            h1 = self.conv1(x)
            h1 = self.conv1n(h1)
            h1 = relu.relu(h1)
            outs.append(h1)

        h3 = relu.relu(self.proj3n(self.proj3(x)))
        h3 = relu.relu(self.conv3n(self.conv3(h3)))
        outs.append(h3)

        h33 = relu.relu(self.proj33n(self.proj33(x)))
        h33 = relu.relu(self.conv33an(self.conv33a(h33)))
        h33 = relu.relu(self.conv33bn(self.conv33b(h33)))
        outs.append(h33)

        if self.pooltype == 'max':
            p = max_pooling_2d.max_pooling_2d(x, 3, stride=self.stride, pad=1,
                                              cover_all=False)
        else:
            p = average_pooling_2d.average_pooling_2d(x, 3, stride=self.stride,
                                                      pad=1)
        if self.proj_pool is not None:
            p = relu.relu(self.poolpn(self.poolp(p)))
        outs.append(p)

        y = concat.concat(outs, axis=1)
        return y
