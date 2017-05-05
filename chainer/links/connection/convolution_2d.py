import numpy

from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link


class Convolution2D(link.Link):

    """Two-dimensional convolutional layer.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

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

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None):
        super(Convolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels

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
            x, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
