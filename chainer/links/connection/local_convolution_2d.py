from chainer.functions.connection import local_convolution_2d
from chainer import initializers
from chainer import link
from chainer import variable


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


def _conv_output_length(input_length, filter_size, stride):
    output_length = input_length - filter_size + 1
    return output_length


class LocalConvolution2D(link.Link):

    """Two-dimensional local convolutional layer.

    This link wraps the :func:`~chainer.functions.local_convolution_2d`
    function and holds the filter weight and bias array as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If either
            in_channels or in_size is ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays
        in_size (int or pair of ints): Size of each image channel
            ``in_size=k`` and ``in_size=(k,k)`` are equivalent. If either
            in_channels or in_size is ``None``, parameter initialization will
            be deferred until the first forward data pass when the size will be
            determined.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 6.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 3.

    .. seealso::
       See :func:`chainer.functions.local_convolution_2d`.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.
    """

    def __init__(self, in_channels, out_channels, in_size=None, ksize=None,
                 stride=1, nobias=False, initialW=None, initial_bias=None,
                 **kwargs):
        super(LocalConvolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.nobias = nobias
        self.out_channels = out_channels
        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer)

            if in_channels is not None and in_size is not None:
                self._initialize_params(in_channels, _pair(in_size))

    def _initialize_params(self, in_channels, in_size):
        kh, kw = _pair(self.ksize)
        ih, iw = _pair(in_size)
        oh = _conv_output_length(ih, kh, self.stride[0])
        ow = _conv_output_length(iw, kw, self.stride[1])
        W_shape = (self.out_channels, oh, ow, in_channels, kh, kw)
        bias_shape = (self.out_channels, oh, ow,)
        self.W.initialize(W_shape)
        if not self.nobias:
            self.b.initialize(bias_shape)

    def forward(self, x):
        """Applies the local convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.W.array is None:
            self._initialize_params(x.shape[1], x.shape[2:])
        return local_convolution_2d.local_convolution_2d(
            x, self.W, self.b, self.stride)
