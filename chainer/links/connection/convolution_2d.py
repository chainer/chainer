import math

import numpy

from chainer.functions.connection import convolution_2d
from chainer import link
from chainer import variable


class Convolution2D(link.Link):

    """Two-dimensional convolution layer.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or (int, int)): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If True, then this link does not use the bias term.
        use_cudnn (bool): If True, then this link uses CuDNN if available.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
        dtype (numpy.dtype): Type to use in computing.

    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.

    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None,
                 dtype=numpy.float32):
        super(Convolution2D, self).__init__()
        self._conv_arg = (stride, pad, use_cudnn)

        dtype = numpy.dtype(dtype)
        kh, kw = _pair(ksize)

        W_shape = (out_channels, in_channels, kh, kw)
        if initialW is not None:
            assert initialW.shape == W_shape
            self.params['W'] = variable.Variable(initialW)
        else:
            std = wscale * math.sqrt(1. / (kh * kw * in_channels))
            self.params['W'] = variable.Variable(numpy.random.normal(
                0, std, W_shape).astype(dtype))

        if initial_bias is not None:
            assert initial_bias.shape == (out_channels,)
            self.params['b'] = variable.Variable(initial_bias)
        elif not nobias:
            self.params['b'] = variable.Variable(
                numpy.repeat(dtype.type(bias), out_channels))

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        W = self.params['W']
        b = self.params.get('b', None)
        return convolution_2d.convolution_2d(x, W, b, *self._conv_arg)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)
