import six

from chainer.functions.array import concat
from chainer import link
from chainer.links.connection import convolution_2d


class SeparableConvolution2D(link.ChainList):

    """Separable convolutional layer.

    This is an "separableconv" layer from the Rigid-Motion Scattering For
    Image Classification paper.
    Google Xception architecture uses this network, and default configuration
    of this layer follows recommendation by Xception.
    This layer is a depthwise (i.e. input channel wise) convolution layer
    followed by 1x1 pointwise convolution layer.
    In depthwise conv, filters with different weight will be applied for each
    input channel.

    Note that it does not apply the activation function to the output of the
    pointwise convolution layers.

    Args:
        in_channels (int or None): Number of channels of input arrays.
        mid_channels (int): Number of channels of output of depthwise
            conv layer. It must equals the number of input channels of
            pointwise conv layers.
        out_channels (int): Number of output channels. The i-th
            integer indicates the number of filters of the i-th convolution.
        depthwise_conv (~chainer.Links.Convolution2D): Definition of depthwise
            conv layer. If `None`, default configuration recommended by the
            author of Xception will be used.
        pointwise_conv (~chainer.Links.Convolution2D): Definition of pointwise
            conv layer. If `None`, default configuration recommended by the
            author of Xception will be used.

    See: `Rigid-Motion Scattering For Image Classification
         <http://www.cmapx.polytechnique.fr/~sifre/research/phd_sifre.pdf>`.
         `Xception: Deep Learning with Depthwise Separable Convolutions
         <https://arxiv.org/abs/1610.02357>`.
    """

    def __init__(self, in_channels, mid_channels, out_channels=None,
                 pointwise_conv=None, depthwise_conv=None):
        if out_channels is None:
            out_channels = mid_channels
        if depthwise_conv is None:
            depthwise_conv = depthwise_convolution_2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      ksize=1)
        if pointwise_conv is None:
            pointwise_conv = convolution_2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            ksize=3)
        assert depthwise_conv.out_channels == pointwise_conv.in_channels
        convs = [depthwise_conv]
        for i in six.moves.range(mid_channels):
            convs.append(pointwise_conv.copy())
        super(SeparableConvolution2D, self).__init__(*convs)

    def __call__(self, x):
        """Computes the output of the SeparableConv layer.

        Args:
            x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
                Input image.

        Returns:
            ~chainer.Variable:
                Output of the SeparableConv layer.
                It is 4D tensor.

        """
        x = self[0](x)
        ys = []
        for i in six.moves.range(len(self) - 1):
            ys.append(self[i](x[i]))
        return concat.concat(*ys, axis=1)
