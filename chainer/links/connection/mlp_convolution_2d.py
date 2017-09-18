from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.utils import argument


class MLPConvolution2D(link.ChainList):

    """__init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, activation=relu.relu, conv_init=None, bias_init=None)

    Two-dimensional MLP convolution layer of Network in Network.

    This is an "mlpconv" layer from the Network in Network paper. This layer
    is a two-dimensional convolution layer followed by 1x1 convolution layers
    and interleaved activation functions.

    Note that it does not apply the activation function to the output of the
    last 1x1 convolution layer.

    Args:
        in_channels (int or None): Number of channels of input arrays.
            If it is ``None`` or omitted, parameter initialization will be
            deferred until the first forward data pass at which time the size
            will be determined.
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
        conv_init: An initializer of weight matrices
            passed to the convolution layers. This option must be specified as
            a keyword argument.
        bias_init: An initializer of bias vectors
            passed to the convolution layers. This option must be specified as
            a keyword argument.

    .. note:
        From v2, `conv_init` and `bias_init` arguments must be specified as
        keyword arguments only. We impose this restriction to forbid
        users to assume the API for v1 and specify `wscale` option, 
        that had been between `activation` and `conv_init` arguments in v1.

    See: `Network in Network <https://arxiv.org/abs/1312.4400v3>`_.

    Attributes:
        activation (function): Activation function.

    """  # NOQA

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 activation=relu.relu, *args, **kwargs):

        # If `args` is not empty, users assume the API for v1 and
        # specify `wscale` as a positonal argument, which we want
        # to detect and forbid with an explicit error message.
        msg = ('wscale is not supported anymore. '
               'Use conv_init and bias_init argument to change '
               'the scale of initial parameters.')
        if args:
            raise TypeError(msg)
        argument.check_unexpected_kwargs(kwargs, wscale=msg)
        conv_init, bias_init = argument.parse_kwargs(
            kwargs, ('conv_init', None), ('bias_init', None))

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        assert len(out_channels) > 0
        convs = [convolution_2d.Convolution2D(
            in_channels, out_channels[0], ksize, stride, pad,
            initialW=conv_init, initial_bias=bias_init)]
        for n_in, n_out in zip(out_channels, out_channels[1:]):
            convs.append(convolution_2d.Convolution2D(
                n_in, n_out, 1, initialW=conv_init,
                initial_bias=bias_init))
        super(MLPConvolution2D, self).__init__(*convs)
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
