from chainer.functions import deformable_convolution_2d_sampler
from chainer import initializers
from chainer.initializers import constant
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer import variable


class DeformableConvolution2D(link.Chain):
    """Two-dimensional deformable convolutional layer.

    This link wraps the
    convolution layer for offset prediction and
    the :func:`~chainer.functions.deformable_convolution_2d_sampler`
    function.
    This also holds the filter weights and bias vectors of two
    convolution layers as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        offset_nobias (bool): If ``True``, then this link does not use the
            bias term for the first convolution layer.
        offset_initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight of the first convolution layer.
            When it is :class:`numpy.ndarray`, its ``ndim`` should be 4.
        offset_initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias of the first convolution layer.
            If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
        deform_nobias (bool): If ``True``, then this link does not use the
            bias term for the second convolution layer.
        deform_initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight for the second convolution layer.
            When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 4.
        deform_initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias for the second convolution layer.
            If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.

    .. seealso::
       See :func:`chainer.functions.deformable_convolution_2d_sampler`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 offset_nobias=False, offset_initialW=None,
                 offset_initial_bias=None,
                 deform_nobias=False,
                 deform_initialW=None, deform_initial_bias=None):
        super(DeformableConvolution2D, self).__init__()
        kh, kw = _pair(ksize)

        with self.init_scope():
            self.offset_conv = Convolution2D(
                in_channels, 2 * kh * kw, ksize, stride, pad,
                offset_nobias, offset_initialW, offset_initial_bias)
            self.deform_conv = DeformableConvolution2DSampler(
                in_channels, out_channels, ksize, stride, pad,
                deform_nobias, deform_initialW, deform_initial_bias)

    def forward(self, x):
        """Applies the deformable convolution.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the deformable convolution.

        """
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class DeformableConvolution2DSampler(link.Link):
    """Apply a two-dimensional deformable convolution layer using offsets"""

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

    def forward(self, x, offset):
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        return deformable_convolution_2d_sampler(
            x, offset, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
