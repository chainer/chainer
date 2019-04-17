import numpy

from chainer.backends import cuda
from chainer.functions.connection import deconvolution_2d
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable


class Deconvolution2D(link.Link):

    """__init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, \
nobias=False, outsize=None, initialW=None, initial_bias=None, *, dilate=1, \
groups=1)

    Two dimensional deconvolution function.

    This link wraps the :func:`~chainer.functions.deconvolution_2d` function
    and holds the filter weight and bias vector as parameters.

    Deconvolution links can use a feature of cuDNN called autotuning, which
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
        nobias (bool): If ``True``, then this function does not use the bias
            term.
        outsize (tuple): Expected output size of deconvolutional operation.
            It should be pair of height and width :math:`(out_H, out_W)`.
            Default value is ``None`` and the outsize is estimated by
            input size, stride and pad.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 4.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
        dilate (:class:`int` or :class:`tuple` of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (int): The number of groups to use grouped deconvolution. The
            default is one, where grouped deconvolution is not used.

    The filter weight has four dimensions :math:`(c_I, c_O, k_H, k_W)`
    which indicate the number of input channels, output channels,
    height and width of the kernels, respectively.
    The filter weight is initialized with i.i.d. Gaussian random samples, each
    of which has zero mean and deviation :math:`\\sqrt{1/(c_I k_H k_W)}` by
    default.

    The bias vector is of size :math:`c_O`.
    Its elements are initialized by ``bias`` argument.
    If ``nobias`` argument is set to True, then this function does not hold
    the bias parameter.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.cudnn_deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

    .. seealso::
       See :func:`chainer.functions.deconvolution_2d` for the definition of
       two-dimensional convolution.

    .. seealso::
        See :func:`chainer.links.Convolution2D` for the examples of ways to
        give arguments to this link.

    .. admonition:: Example

        There are several ways to make a Deconvolution2D link.

        Let an input vector ``x`` be:

        >>> x = np.arange(1 * 3 * 10 * 10, dtype=np.float32).reshape(
        ...     1, 3, 10, 10)

        1. Give the first three arguments explicitly:

            In this case, all the other arguments are set to the default
            values.

            >>> l = L.Deconvolution2D(3, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 13, 13)

        2. Omit ``in_channels`` or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.Deconvolution2D(7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 13, 13)

            >>> l = L.Deconvolution2D(None, 7, 4)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 13, 13)

            When you omit the first argument, you need to specify the other
            subsequent arguments from ``stride`` as keyword arguments. So the
            below two cases are the same.

            >>> l = L.Deconvolution2D(None, 7, 4, 2, 1)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 20, 20)

            >>> l = L.Deconvolution2D(7, 4, stride=2, pad=1)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 20, 20)

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, outsize=None, initialW=None, initial_bias=None,
                 **kwargs):
        super(Deconvolution2D, self).__init__()

        dilate, groups, = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1),
            deterministic='deterministic argument is not supported anymore. '
            'Use chainer.using_config(\'cudnn_deterministic\', value) '
            'context where value is either `True` or `False`.')

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.outsize = (None, None) if outsize is None else outsize
        self.out_channels = out_channels
        self.groups = int(groups)

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if isinstance(initial_bias, (numpy.ndarray, cuda.ndarray)):
                    assert initial_bias.shape == (out_channels,)
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             'divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             'divisible by the number of groups')
        W_shape = (in_channels, int(self.out_channels / self.groups), kh, kw)
        self.W.initialize(W_shape)

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        return deconvolution_2d.deconvolution_2d(
            x, self.W, self.b, self.stride, self.pad, self.outsize,
            dilate=self.dilate, groups=self.groups)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
