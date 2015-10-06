from chainer.functions.activation import relu
from chainer.links.connection import convolution_2d
from chainer import link


class MLPConvolution2D(link.ListLink):

    """Two-dimensional MLP convolution layer of Network in Network.

    This is an "mlpconv" layer from the Network in Network paper. This layer
    is a two-dimensional convolution layer followed by 1x1 convolution layers
    and interleaved activation functions.

    Args:
        in_channels (int): Number of channels of input arrays.
        out_channels (tuple of ints): Tuple of number of channels. The i-th
            integer indicates the number of filters of the i-th convolution.
        ksize (int or (int, int)): Size of filters (a.k.a. kernels) of the
            first convolution layer. ``ksize=k`` and ``ksize=(k, k)`` are
            equivalent.
        stride (int or (int, int)): Stride of filter applications at the first
            convolution layer. ``stride=s`` and ``stride=(s, s)`` are
            equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays at the
            first convolution layer. ``pad=p`` and ``pad=(p, p)`` are
            equivalent.
        activation (function): Activation function for internal hidden units.
            Note that this function is not applied to the output of this link.
        use_cudnn (bool): If True, then this link uses CuDNN if available.

    See: `Network in Network <http://arxiv.org/abs/1312.4400v3>`.

    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, activation=relu.relu, use_cudnn=True):
        assert len(out_channels) > 0
        super(MLPConvolution2D, self).__init__(
            convolution_2d.Convolution2D(
                in_channels, out_channels[0], ksize, stride, pad,
                wscale=wscale, use_cudnn=use_cudnn),
        )
        for n_in, n_out in zip(out_channels, out_channels[1:]):
            self.append(convolution_2d.Convolution2D(
                n_in, n_out, 1, wscale=wscale))
        self.activation = activation

    def __call__(self, x):
        """Computes the output of the mlpconv layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the mlpconv layer.

        """
        f = self.activation
        for l in self[:-1]:
            x = f(l(x))
        return self[-1](x)
