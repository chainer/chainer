from chainer.functions.connection import convolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer.utils import conv_nd
from chainer import variable


class ConvolutionND(link.Link):
    """N-dimensional convolution layer.

    This link wraps the :func:`~chainer.functions.convolution_nd` function and
    holds the filter weight and bias vector as parameters.

    Convolution links can use a feature of cuDNN called autotuning, which
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
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be :math:`n+2` where :math:`n` is
            the number of spatial dimensions.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should 1.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            ``cover_all`` needs to be ``False`` if you want to use cuDNN.
        dilate (:class:`int` or :class:`tuple` of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d, ..., d)`` are equivalent.
        groups (:class:`int`):
            The number of groups to use grouped convolution.
            The default is one, where grouped convolution is not used.

    .. seealso::
        See :func:`~chainer.functions.convolution_nd` for the definition of
        N-dimensional convolution. See
        :func:`~chainer.functions.convolution_2d` for the definition of
        two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    .. admonition:: Example

        There are several ways to make a ConvolutionND link.

        Let an input vector ``x`` be:

        >>> x = np.arange(2 * 5 * 5 * 5, dtype=np.float32).reshape(
        ...     1, 2, 5, 5, 5)

        1. Give the first four arguments explicitly:

            >>> l = L.ConvolutionND(3, 2, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 2, 2, 2)

        2. Omit ``in_channels`` or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.ConvolutionND(3, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 2, 2, 2)

            >>> l = L.ConvolutionND(3, None, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 2, 2, 2)

            When you omit the second argument, you need to specify the other
            subsequent arguments from ``stride`` as keyword auguments. So the
            below two cases are the same.

            >>> l = L.ConvolutionND(3, 7, 4, stride=1, pad=0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 2, 2, 2)

            >>> l = L.ConvolutionND(3, None, 7, 4, 1, 0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 2, 2, 2)

    """

    def __init__(self, ndim, in_channels, out_channels, ksize=None, stride=1,
                 pad=0, nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):
        super(ConvolutionND, self).__init__()

        if ksize is None:
            out_channels, ksize, in_channels = \
                in_channels, out_channels, None

        self.out_channels = out_channels
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all
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
                             ' divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             ' divisible by the number of groups')
        W_shape = (
            self.out_channels, int(in_channels / self.groups)) + self.ksize
        self.W.initialize(W_shape)

    @classmethod
    def from_params(cls, W, b=None, stride=1, pad=0, nobias=False, **kwargs):
        """from_params(cls, W, b=None, stride=1, pad=0, \
nobias=False, *, cover_all=False, dilate=1, groups=1)

        Initialize a :class:`~chainer.links.ConvolutionND` with given
        parameters.

        This method uses ``W`` and optional ``b`` to initialize an :math:`N` D
        convolution layer.

        Args:
            W (:class:`~chainer.Variable` or :ref:`ndarray`):
                The weight parameter.
            b (:class:`~chainer.Variable`, :ref:`ndarray`, or None):
                The weight parameter.
            ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
                ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
            stride (int or tuple of ints): Stride of filter application.
                ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
            pad (int or tuple of ints): Spatial padding width for input arrays.
                ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
            nobias (bool):
                If ``True``, then this function does not use the bias.
            cover_all (bool): If ``True``, all spatial locations are convoluted
                into some output pixels. It may make the output size larger.
                ``cover_all`` needs to be ``False`` if you want to use cuDNN.
            dilate (:class:`int` or :class:`tuple` of :class:`int` s):
                Dilation factor of filter applications.
                ``dilate=d`` and ``dilate=(d, d, ..., d)`` are equivalent.
            groups (:class:`int`):
                The number of groups to use grouped convolution.
                The default is one, where grouped convolution is not used.
        """
        # TODO(crcrpar): Support the below conditions.
        # - W (and b) of cupy on non-default GPUs like id=1.
        # - W (and b) of chainerx on cuda.
        cover_all, dilate, groups = argument.parse_kwargs(
            kwargs, ('cover_all', False), ('dilate', 1), ('groups', 1))
        out_channels, _in_channels, *ksize = W.shape
        in_channels = _in_channels * groups
        if b is not None:
            if out_channels != b.size:
                raise ValueError(
                    '`out_channels` does not match the size of `b`')

        link = cls(
            len(ksize), in_channels, out_channels, ksize, stride, pad, nobias,
            initialW=variable.as_array(W), initial_bias=variable.as_array(b),
            cover_all=cover_all, dilate=dilate, groups=groups)
        return link

    def forward(self, x):
        """Applies N-dimensional convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of convolution.

        """
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        return convolution_nd.convolution_nd(
            x, self.W, self.b, self.stride, self.pad, cover_all=self.cover_all,
            dilate=self.dilate, groups=self.groups)


class Convolution1D(ConvolutionND):
    """1-dimensional convolution layer.

    .. note::

        This link wraps :class:`~chainer.links.ConvolutionND` by giving 1 to
        the first argument ``ndim``, so see the details of the behavior in
        the documentation of :class:`~chainer.links.ConvolutionND`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):
        super(Convolution1D, self).__init__(
            1, in_channels, out_channels, ksize, stride, pad, nobias, initialW,
            initial_bias, cover_all, dilate, groups)


class Convolution3D(ConvolutionND):
    """3-dimensional convolution layer.

    .. note::

        This link wraps :class:`~chainer.links.ConvolutionND` by giving 3 to
        the first argument ``ndim``, so see the details of the behavior in
        the documentation of :class:`~chainer.links.ConvolutionND`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):
        super(Convolution3D, self).__init__(
            3, in_channels, out_channels, ksize, stride, pad, nobias, initialW,
            initial_bias, cover_all, dilate, groups)
