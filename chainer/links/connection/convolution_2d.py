import chainer
from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link
from chainer import memory_layouts
from chainer.utils import argument
from chainer import variable


class Convolution2D(link.Link):

    """__init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, \
nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1)

    Two-dimensional convolutional layer.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

    Convolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    Args:
        in_channels (int or None): Number of channels of input arrays.
            If ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
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
        dilate (int or pair of ints):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (:class:`int`): Number of groups of channels. If the number
            is greater than 1, input tensor :math:`W` is divided into some
            blocks by this value channel-wise. For each tensor blocks,
            convolution operation will be executed independently. Input channel
            size ``in_channels`` and output channel size ``out_channels`` must
            be exactly divisible by this value.

    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    .. admonition:: Example

        There are several ways to make a Convolution2D link.

        Let an input vector ``x`` be:

        >>> x = np.arange(1 * 3 * 10 * 10, dtype=np.float32).reshape(
        ...     1, 3, 10, 10)

        1. Give the first three arguments explicitly:

            >>> l = L.Convolution2D(3, 7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

        2. Omit ``in_channels`` or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.Convolution2D(7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

            >>> l = L.Convolution2D(None, 7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

            When you omit the first argument, you need to specify the other
            subsequent arguments from ``stride`` as keyword auguments. So the
            below two cases are the same.

            >>> l = L.Convolution2D(7, 5, stride=1, pad=0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

            >>> l = L.Convolution2D(None, 7, 5, 1, 0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(Convolution2D, self).__init__()

        dilate, groups = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1),
            deterministic='deterministic argument is not supported anymore. '
            'Use chainer.using_config(\'cudnn_deterministic\', value) '
            'context where value is either `True` or `False`.')

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.cudnn_fast = chainer.get_compute_mode() == 'cudnn_fast'
        if self.cudnn_fast:
            x_layout = memory_layouts.CUDNN_CHANNEL_LAST_X
            w_layout = memory_layouts.CUDNN_CHANNEL_LAST_W
        else:
            x_layout = memory_layouts.CUDNN_CHANNEL_FIRST_X
            w_layout = memory_layouts.CUDNN_CHANNEL_FIRST_W

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = int(groups)
        self.x_layout = x_layout

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer, layout=w_layout)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    @property
    def printable_specs(self):
        specs = [
            ('in_channels', self.in_channels),
            ('out_channels', self.out_channels),
            ('ksize', self.ksize),
            ('stride', self.stride),
            ('pad', self.pad),
            ('nobias', self.b is None),
            ('dilate', self.dilate),
            ('groups', self.groups),
        ]
        for spec in specs:
            yield spec

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             ' divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             ' divisible by the number of groups')
        W_shape = (self.out_channels, int(in_channels / self.groups), kh, kw)
        self.W.initialize(W_shape)

    @classmethod
    def from_params(cls, W, b=None, stride=1, pad=0, nobias=False, **kwargs):
        """from_params(cls, W, b=None, stride=1, pad=0, \
nobias=False, *, dilate=1, groups=1)

        Initialize a :class:`~chainer.links.Convolution2D` with given
        parameters.

        This method uses ``W`` and optional ``b`` to initialize
        a 2D convolution layer.

        Args:
            W (:class:`~chainer.Variable` or :ref:`ndarray`):
                The weight parameter.
            b (:class:`~chainer.Variable`, :ref:`ndarray`, or ``None``):
                The bias parameter.
            stride (int or pair of ints): Stride of filter applications.
                ``stride=s`` and ``stride=(s, s)`` are equivalent.
            pad (int or pair of ints): Spatial padding width for input arrays.
                ``pad=p`` and ``pad=(p, p)`` are equivalent.
            nobias (bool): If ``True``, then this link does not use
                the bias term in spite of whether ``b`` is given or not.
            dilate (int or pair of ints):
                Dilation factor of filter applications.
                ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
            groups (:class:`int`): Number of groups of channels. If the number
                is greater than 1, input tensor :math:`W` is divided into some
                blocks by this value channel-wise. For each tensor blocks,
                convolution operation will be executed independently.
                Input channel size ``in_channels`` and output channel size
                ``out_channels`` must be exactly divisible by this value.
        """
        # TODO(crcrpar): Support the below conditions.
        # - W (and b) of cupy on non-default GPUs like id=1.
        # - W (and b) of chainerx on cuda.
        dilate, groups = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1))
        out_channels, _in_channels, kw, kh = W.shape
        in_channels = _in_channels * groups
        if b is not None:
            if out_channels != b.size:
                raise ValueError(
                    '`out_channels` does not match the size of `b`')

        link = cls(
            in_channels, out_channels, (kw, kh), stride, pad, nobias,
            initialW=variable.as_array(W), initial_bias=variable.as_array(b),
            dilate=dilate, groups=groups)
        return link

    def forward(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        x = chainer.as_variable(x)
        assert x.layout == self.x_layout
        # self.W can be a Variable instead of Parameter: #8462
        # TODO(niboshi): Use Parameter.is_initialized.
        if self.W.raw_array is None:
            _, c, _, _ = memory_layouts.get_semantic_shape(
                x, assumed_layout=self.x_layout)
            self._initialize_params(c)
        return convolution_2d.convolution_2d(
            x, self.W, self.b, self.stride, self.pad, dilate=self.dilate,
            groups=self.groups, cudnn_fast=self.cudnn_fast)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
