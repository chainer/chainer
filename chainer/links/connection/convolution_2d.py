import numpy

from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link


class Convolution2D(link.Link):

    """Two-dimensional convolutional layer.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If it is
            ``None`` or omitted, parameter initialization will be deferred
            until the first forward data pass at which time the size will be
            determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (callable): Weight initializer.
            It should be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
            If it is `numpy.ndarray`, the array is used as initial
            weight value.
        initial_bias (1-D array): Initial bias value.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        deterministic (bool): The output of this link can be
            non-deterministic when it uses cuDNN.
            If this option is ``True``, then it forces cuDNN to use
            a deterministic algorithm. This option is only available for
            cuDNN version >= v4.

    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.


    .. admonition:: Example

        There are several ways to make a Convolution2D link.

        Let an input vector ``x`` be:

        >>> x = np.arange(1 * 3 * 10 * 10).astype('f').reshape(1, 3, 10, 10)

        1. Give the first three arguments explicitly:

            >>> l = L.Convolution2D(3, 7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

        2. Omit ``in_channels``:

            The below two cases are the same.

            >>> l = L.Convolution2D(7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

            >>> l = L.Convolution2D(None, 7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

        3. Give arguments other than the first three arguments besides omitting
            ``in_channels``:

            When you omit the first argument, you need to specify the other
            subsequent arguments from ``stride`` as keyword auguments. So the
            below two cases are the same.

            >>> l = L.Convolution2D(None, 7, 5, 1, 0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

            >>> l = L.Convolution2D(7, 5, stride=1, pad=0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 deterministic=False):
        super(Convolution2D, self).__init__()

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels
        self.deterministic = deterministic

        if initialW is None:
            initialW = initializers.HeNormal(1. / numpy.sqrt(2))
        self.add_param('W', initializer=initializers._get_initializer(
            initialW))
        if in_channels is not None:
            self._initialize_params(in_channels)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = initializers.Constant(0)
            bias_initilizer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=bias_initilizer)

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
            x, self.W, self.b, self.stride, self.pad,
            deterministic=self.deterministic)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
