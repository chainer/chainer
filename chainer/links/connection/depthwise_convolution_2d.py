import numpy

from chainer.functions.connection import depthwise_convolution_2d
from chainer import initializers
from chainer import link
from chainer import variable


class DepthwiseConvolution2D(link.Link):

    """Two-dimensional depthwise convolutional layer.

    This link wraps the :func:`~chainer.functions.depthwise_convolution_2d`
    function and holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        channel_multiplier (int): Channel multiplier number. Number of output
            arrays equal ``in_channels * channel_multiplier``.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 4.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.

    .. seealso::
       See :func:`chainer.functions.depthwise_convolution_2d`.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, in_channels, channel_multiplier, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None):
        super(DepthwiseConvolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.channel_multiplier = channel_multiplier
        self.nobias = nobias

        if initialW is None:
            initialW = initializers.HeNormal(1. / numpy.sqrt(2))

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
        W_shape = (self.channel_multiplier, in_channels, kh, kw)
        self.W.initialize(W_shape)
        if self.b is not None:
            self.b.initialize(self.channel_multiplier * in_channels)

    def __call__(self, x):
        """Applies the depthwise convolution layer.

        Args:
            x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
                Input image.

        Returns:
            ~chainer.Variable: Output of the depthwise convolution.

        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return depthwise_convolution_2d.depthwise_convolution_2d(
            x, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
