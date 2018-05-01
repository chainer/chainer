import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.connection import convolution_2d
from chainer.links.connection.convolution_2d import Convolution2D


class SpectralNormalizationConvolution2D(Convolution2D):
    """Two-dimensional convolutional layer with spectral normalization.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            datasets pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        use_gamma (bool): If true, apply scalar multiplication to the
            normalized weight (i.e. reparameterize).
        Ip (int): The number of power iteration for calculating the spcetral
            norm of the weights.
        factor (float) : constant factor to adjust spectral norm of W_bar.

    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        W_bar (~chainer.Variable): Spectrally normalized weight parameter.
        b (~chainer.Variable): Bias parameter.
        u (~numpy.array): Current estimation of the right largest singular
                          vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
        (optional) factor (float): Constant factor to adjust spectral norm of
                                   W_bar.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 use_gamma=False, Ip=1, factor=None, dtype=np.float32):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SpectralNormalizationConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, initialW, initial_bias)
        self.u = np.random.normal(size=(1, out_channels)).astype(
             dtype=dtype)
        self.register_persistent('u')

    @property
    def W_bar(self):
        """Spectrally normalized weight"""
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = _max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = F.broadcast_to(sigma.reshape((1, 1, 1, 1)), self.W.shape)
        if chainer.config.train:
            # Update estimated 1st singular vector
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return F.broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SpectralNormalizationConvolution2D, self)._initialize_params(
            in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

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
            x, self.W_bar, self.b, self.stride, self.pad, dilate=self.dilate,
            groups=self.groups)


def _l2normalize(v, eps=1e-12):
    xp = cuda.get_array_module(v)
    if xp is np:
        return v / (np.linalg.norm(v) + eps)
    else:
        norm = cuda.reduce('T x', 'T out',
                           'x * x', 'a + b', 'out = sqrt(a)', 0,
                           'norm_sn')
        div = cuda.elementwise('T x, T norm, T eps',
                               'T out',
                               'out = x / (norm + eps)',
                               'div_sn')
        return div(v, norm(v), eps)


def _max_singular_value(W, u=None, Ip=1):
    """Apply power iteration for the weight parameter"""
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be a "
                         "positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = F.sum(F.linear(_u, F.transpose(W)) * _v)
    return sigma, _u, _v
