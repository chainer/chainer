import numpy

from chainer.functions.connection import convolution_3d
from chainer import link


class Convolution3D(link.Link):

    """Three-dimensional convolutional layer.
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None):
        kh, kw, kd = _triplet(ksize)
        self.stride = _triplet(stride)
        self.pad = _triplet(pad)
        self.use_cudnn = use_cudnn

        W_shape = (out_channels, in_channels, kh, kw, kd)
        super(Convolution3D, self).__init__(W=W_shape)

        if initialW is not None:
            self.W.data[...] = initialW
        else:
            std = wscale * numpy.sqrt(1. / (kh * kw * in_channels))
            self.W.data[...] = numpy.random.normal(0, std, W_shape)

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_channels)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        return convolution_3d.convolution_3d(
            x, self.W, self.b, self.stride, self.pad, self.use_cudnn)


def _triplet(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x, x
