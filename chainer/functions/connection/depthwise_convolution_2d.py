import chainer


def depthwise_convolution_2d(x, W, b=None, stride=1, pad=0):
    """Two-dimensional depthwise convolution function.

    This is an implementation of two-dimensional depthwise convolution.
    It takes two or three variables: the input image ``x``, the filter weight
    ``W``, and optionally, the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input.
    - :math:`c_M` is the channel multiplier.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable of shape :math:`(n, c_I, h, w)`.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight variable of shape :math:`(c_M, c_I, k_H, k_W)`.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias variable of length :math:`c_M * c_I` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.


    Returns:
        ~chainer.Variable:
            Output variable. Its shape is :math:`(n, c_I * c_M, h_O, w_O)`.

    Like ``Convolution2D``, ``DepthwiseConvolution2D`` function computes
    correlations between filters and patches of size :math:`(k_H, k_W)` in
    ``x``.
    But unlike ``Convolution2D``, ``DepthwiseConvolution2D`` does not add up
    input channels of filters but concatenates them.
    For that reason, the shape of outputs of depthwise convolution are
    :math:`(n, c_I * c_M, h_O, w_O)`, :math:`c_M` is called channel_multiplier.

    :math:`(h_O, w_O)` is determined by the equivalent equation of
    ``Convolution2D``.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    See: `L. Sifre. Rigid-motion scattering for image classification
    <https://www.di.ens.fr/data/publications/papers/phd_sifre.pdf>`_

    .. seealso::

        :class:`~chainer.links.DepthwiseConvolution2D`
        to manage the model parameters ``W`` and ``b``.

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (2, 3, 4, 7))
        >>> W = np.random.uniform(0, 1, (2, 3, 3, 3))
        >>> b = np.random.uniform(0, 1, (6,))
        >>> y = F.depthwise_convolution_2d(x, W, b)
        >>> y.shape
        (2, 6, 2, 5)

    """
    multiplier, in_channels, kh, kw = W.shape
    F = chainer.functions
    W = F.transpose(W, (1, 0, 2, 3))
    W = F.reshape(W, (multiplier * in_channels, 1, kh, kw))
    return F.convolution_2d(x, W, b, stride, pad, groups=in_channels)
