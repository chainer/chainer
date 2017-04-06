import numpy

from chainer.functions.connection import convolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd


class ConvolutionND(link.Link):
    """N-dimensional convolution layer.

    This link wraps the :func:`~chainer.functions.convolution_nd` function and
    holds the filter weight and bias vector as parameters.

    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints): Stride of filter application.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (callable): Weight initializer.
            It should be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
            If it is `numpy.ndarray`, the array is used as initial
            weight value.
        initial_bias (callable): Bias initializer.
            It should be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If ``None``, the default initializer is used.
            If it is `numpy.ndarray`, the array is used as initial bias value.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            ``cover_all`` needs to be ``False`` if you want to use cuDNN.

    .. seealso::
        See :func:`~chainer.functions.convolution_nd` for the definition of
        N-dimensional convolution. See
        :func:`~chainer.functions.convolution_2d` for the definition of
        two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False):
        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all

        super(ConvolutionND, self).__init__()

        W_shape = (out_channels, in_channels) + ksize
        if initialW is None:
            initializers.HeNormal(1. / numpy.sqrt(2))
        initialW = initializers._get_initializer(initialW)
        self.add_param('W', W_shape, initializer=initialW)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = initializers.Constant(0)
            initial_bias = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=initial_bias)

    def __call__(self, x):
        """Applies N-dimensional convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of convolution.

        """
        return convolution_nd.convolution_nd(
            x, self.W, self.b, self.stride, self.pad, cover_all=self.cover_all)
