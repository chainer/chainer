import math

import chainer
from chainer import cuda
from chainer.functions import deformable_convolution_2d_sampler
from chainer.functions.array import pad
from chainer import initializers
from chainer.initializers import constant
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D


class DeformableConvolution2D(link.Chain):
    """Deformable Convolution 2D.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None,
                 wscale_offset=1, bias_offset=0, nobias_offset=False, use_cudnn_offset=True,
                 initialW_offset=constant.Zero(), initial_bias_offset=None):
        kh, kw = _pair(ksize)

        super(DeformableConvolution2D, self).__init__(
            deform_conv=DeformableConvolution2DSampler(
                in_channels, out_channels, ksize, stride, pad, wscale, bias,
                nobias, use_cudnn, initialW, initial_bias),
            offset_conv=Convolution2D(
                in_channels, 2 * kh * kw, ksize, stride, pad, wscale_offset, bias_offset,
                nobias_offset, use_cudnn_offset, initialW_offset, initial_bias_offset)
        )

    def __call__(self, x):
        offset = self.offset_conv(x)
        n, _, h, w = offset.shape
        return self.deform_conv(x, offset)


class DeformableConvolution2DSampler(link.Link):

    """Apply Two-dimensional deformable convolution layer using offset.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None):
        super(DeformableConvolution2DSampler, self).__init__()

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.use_cudnn = use_cudnn
        self.out_channels = out_channels
        self.initialW = initialW
        self.wscale = wscale

        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, scale=math.sqrt(wscale))
        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_channels)
            if initial_bias is None:
                initial_bias = bias
            initializers.init_weight(self.b.data, initial_bias)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.add_param('W', W_shape, initializer=self._W_initializer)

    def __call__(self, x, offset):
        """Applies the deformable convolution.

        Args:
            x (~chainer.Variable): Input image.
            offset (~chainer.Variable): Offset image.

        Returns:
            ~chainer.Variable: Output of the deformable convolution.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])
        return deformable_convolution_2d_sampler(
            x, offset, self.W, self.b, self.stride, self.pad, use_cudnn=self.use_cudnn)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
