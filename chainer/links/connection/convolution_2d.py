from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable


class Convolution2D(link.Link):

    """__init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None)

    Two-dimensional convolutional layer.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

    .. warning::

        ``deterministic`` argument is not supported anymore since v2.
        Instead, use ``chainer.using_config('cudnn_deterministic', value``
        (value is either ``True`` or ``False``).
        See :func:`chainer.using_config`.

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
        initialW (4-D array): Initial weight value. If ``None``, the default
            initializer is used.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, the bias
            is set to 0.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    .. admonition:: Example

        There are several ways to make a Convolution2D link.

        Let an input vector ``x`` be:

        >>> x = np.arange(1 * 3 * 10 * 10, dtype='f').reshape(1, 3, 10, 10)

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

    """  # NOQA

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(Convolution2D, self).__init__()

        argument.check_unexpected_kwargs(
            kwargs, deterministic="deterministic argument is not "
            "supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) "
            "context where value is either `True` or `False`.")
        argument.assert_kwargs_empty(kwargs)

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels

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
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.W.initialize(W_shape)

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(
            x, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
