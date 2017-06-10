from chainer.functions.connection import deconvolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd
from chainer import variable


class DeconvolutionND(link.Link):
    """N-dimensional deconvolution function.

    This link wraps :func:`~chainer.functions.deconvolution_nd` function and
    holds the filter weight and bias vector as its parameters.

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
        outsize (tuple of ints): Expected output size of deconvolutional
            operation. It should be a tuple of ints that represents the output
            size of each dimension. Default value is ``None`` and the outsize
            is estimated with input size, stride and pad.
        initialW (array): Initial weight array. If ``None``, the default
            initializer is used. May be an
            initializer instance of another value the same with that
            :func:`~chainer.init_weight` function can take.
        initial_bias (array): Initial bias array. If ``None``, the bias vector
            is set to zero. May be an initializer instance of another value
            the same with that :func:`~chainer.init_weight` function can take.

    .. seealso::
       :func:`~chainer.functions.deconvolution_nd`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, outsize=None,
                 initialW=None, initial_bias=None):
        super(DeconvolutionND, self).__init__()

        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.outsize = outsize

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer,
                                        (in_channels, out_channels) + ksize)
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                initial_bias = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(initial_bias, out_channels)

    def __call__(self, x):
        return deconvolution_nd.deconvolution_nd(
            x, self.W, b=self.b, stride=self.stride, pad=self.pad,
            outsize=self.outsize)
