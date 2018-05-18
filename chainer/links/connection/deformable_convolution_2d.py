from chainer.functions import deformable_convolution_2d_sampler
from chainer import initializers
from chainer.initializers import constant
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer import variable


class DeformableConvolution2D(link.Chain):
    """Deformable Convolution 2D.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 offset_nobias=False,
                 offset_initialW=None, offset_initial_bias=None):
        super(DeformableConvolution2D, self).__init__()
        kh, kw = _pair(ksize)

        with self.init_scope():
            self.deform_conv = DeformableConvolution2DSampler(
                in_channels, out_channels, ksize, stride, pad,
                nobias, initialW, initial_bias)
            self.offset_conv = Convolution2D(
                in_channels, 2 * kh * kw, ksize, stride, pad,
                offset_nobias, offset_initialW, offset_initial_bias)

    def __call__(self, x):
        """Applies the deformable convolution.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the deformable convolution.

        """
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class DeformableConvolution2DSampler(link.Link):
    """Apply Two-dimensional deformable convolution layer using offset"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None):
        super(DeformableConvolution2DSampler, self).__init__()

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels
        self.initialW = initialW

        if initialW is None:
            initialW = constant.Zero()

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = initializers.Constant(0)
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer)

        if in_channels is not None:
            self._initialize_params(in_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.W.initialize(W_shape)
        if self.b is not None:
            self.b.initialize(self.out_channels)

    def __call__(self, x, offset):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return deformable_convolution_2d_sampler(
            x, offset, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
