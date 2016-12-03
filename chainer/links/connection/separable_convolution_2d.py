import six

from chainer.functions.array import concat
from chainer import link
from chainer.links.connection import convolution_2d


class SeparableConvolution2D(link.ChainList):

    """Separable convolutional layer.

    Args:
        in_channels (int or None): Number of channels of input arrays.
        out_channels (tuple of ints): Tuple of number of channels. The i-th
            integer indicates the number of filters of the i-th convolution.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels) of the
            first convolution layer. ``ksize=k`` and ``ksize=(k, k)`` are
            equivalent.
        stride (int or pair of ints): Stride of filter applications at the
            first convolution layer. ``stride=s`` and ``stride=(s, s)`` are
            equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays at
            the first convolution layer. ``pad=p`` and ``pad=(p, p)`` are
            equivalent.
        activation (function): Activation function for internal hidden units.
            Note that this function is not applied to the output of this link.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.
        conv_init: An initializer of weight matrices
            passed to the convolution layers.
        bias_init: An initializer of bias vectors
            passed to the convolution layers.

    See: `Network in Network <http://arxiv.org/abs/1312.4400v3>`.

    Attributes:
        pointwise_conv (Convolution2D): Activation function.
        depthwise_conv (Convolution2D): Activation function.

    """

    def __init__(self, in_channels, mid_channels, out_channels=None,
                 pointwise_conv=None, depthwise_conv=None):
        if out_channels is None:
            out_channels = mid_channels
        if pointwise_conv is None:
            pointwise_conv = convolution_2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            ksize=1)
        if depthwise_conv is None:
            depthwise_conv = convolution_2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            ksize=3,
                                            )
        assert pointwise_conv.out_channels == depthwise_conv.in_channels
        convs = [pointwise_conv]
        for i in six.moves.range(mid_channels):
            convs.append(depthwise_conv.copy())
        super(SeparableConvolution2D, self).__init__(*convs)

    def __call__(self, x):
        """Computes the output of the SeparableConv layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable:
                Output of the SeparableConv layer.
                It is 4D tensor.

        """
        x = self[0](x)
        ys = []
        for depth_conv2d in self[1:]:
            ys.append(depth_conv2d(x))
        return concat.concat(*ys, axis=1)
