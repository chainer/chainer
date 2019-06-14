from chainer.functions.connection import deconvolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd
from chainer import variable


class DeconvolutionND(link.Link):
    """N-dimensional deconvolution function.

    This link wraps :func:`~chainer.functions.deconvolution_nd` function and
    holds the filter weight and bias vector as its parameters.

    Deconvolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of channels of input arrays.
            If ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints): Stride of filter application.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        nobias (bool): If ``True``, then this function does not use the bias.
        outsize (tuple of ints): Expected output size of deconvolutional
            operation. It should be a tuple of ints that represents the output
            size of each dimension. Default value is ``None`` and the outsize
            is estimated with input size, stride and pad.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be :math:`n+2` where :math:`n` is
            the number of spatial dimensions.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should 1.
        dilate (:class:`int` or :class:`tuple` of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d, ..., d)`` are equivalent.
        groups (:class:`int`):
            The number of groups to use grouped convolution.
            The default is one, where grouped convolution is not used.

    .. seealso::
       :func:`~chainer.functions.deconvolution_nd`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    .. admonition:: Example

        There are several ways to make a DeconvolutionND link.

        Let an input vector ``x`` be:

        >>> x = np.arange(2 * 5 * 5 * 5, dtype=np.float32).reshape(
        ...     1, 2, 5, 5, 5)

        1. Give the first four arguments explicitly:

            >>> l = L.DeconvolutionND(3, 2, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 8, 8, 8)

        2. Omit ``in_channels`` or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.DeconvolutionND(3, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 8, 8, 8)

            >>> l = L.DeconvolutionND(3, None, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 8, 8, 8)

            When you omit the second argument, you need to specify the other
            subsequent arguments from ``stride`` as keyword auguments. So the
            below two cases are the same.

            >>> l = L.DeconvolutionND(3, 7, 4, stride=2, pad=1)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 10, 10, 10)

            >>> l = L.DeconvolutionND(3, None, 7, 4, 2, 1)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 10, 10, 10)

    """

    def __init__(self, ndim, in_channels, out_channels, ksize=None, stride=1,
                 pad=0, nobias=False, outsize=None, initialW=None,
                 initial_bias=None, dilate=1, groups=1):
        super(DeconvolutionND, self).__init__()

        if ksize is None:
            out_channels, ksize, in_channels = \
                in_channels, out_channels, None

        self.out_channels = out_channels
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.outsize = outsize
        self.dilate = conv_nd.as_tuple(dilate, ndim)
        self.groups = int(groups)

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                initial_bias = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(initial_bias, out_channels)

    def _initialize_params(self, in_channels):
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             'divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             'divisible by the number of groups')
        W_shape = (
            in_channels, int(self.out_channels / self.groups)) + self.ksize
        self.W.initialize(W_shape)

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        return deconvolution_nd.deconvolution_nd(
            x, self.W, b=self.b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, dilate=self.dilate, groups=self.groups)


class Deconvolution1D(DeconvolutionND):
    """1-dimensional deconvolution layer.

    .. note::

        This link wraps :class:`~chainer.links.DeconvolutionND` by giving 1 to
        the first argument ``ndim``, so see the details of the behavior in
        the documentation of :class:`~chainer.links.DeconvolutionND`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, outsize=None, initialW=None, initial_bias=None,
                 dilate=1, groups=1):
        super(Deconvolution1D, self).__init__(
            1, in_channels, out_channels, ksize, stride, pad, nobias, outsize,
            initialW, initial_bias, dilate, groups)


class Deconvolution3D(DeconvolutionND):
    """3-dimensional deconvolution layer.

    .. note::

        This link wraps :class:`~chainer.links.DeconvolutionND` by giving 3 to
        the first argument ``ndim``, so see the details of the behavior in
        the documentation of :class:`~chainer.links.DeconvolutionND`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, outsize=None, initialW=None, initial_bias=None,
                 dilate=1, groups=1):
        super(Deconvolution3D, self).__init__(
            3, in_channels, out_channels, ksize, stride, pad, nobias, outsize,
            initialW, initial_bias, dilate, groups)
