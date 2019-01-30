from chainer.functions.connection import convolution_2d


def dilated_convolution_2d(x, W, b=None, stride=1, pad=0, dilate=1,
                           cover_all=False):
    """Two-dimensional dilated convolution function.

    This is an implementation of two-dimensional dilated convolution
    in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    .. note::
       You can also perform dilated convolution by passing ``dilate``
       argument to :class:`chainer.functions.convolution_2d`.
       The functionality is the same.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output,
      respectively.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable of shape :math:`(n, c_I, h, w)`.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight variable of shape :math:`(c_O, c_I, k_H, k_W)`.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias variable of length :math:`c_O` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        dilate (int or pair of ints): Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.

    Returns:
        ~chainer.Variable: Output variable.

    The two-dimensional dilated convolution function is defined as follows.
    Then the ``DilatedConvolution2D`` function computes correlations
    between filters and patches of size :math:`(k_H, k_W)` in ``x``.
    Patches here are extracted at intervals of the dilation factor.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at intervals of the dilation factor and at positions
    shifted by multiples of ``stride`` from the first position ``-pad`` for
    each spatial axis. The right-most (or bottom-most) patches do not run over
    the padded spatial size.

    Let :math:`(s_Y, s_X)` be the stride of filter application,
    :math:`(p_H, p_W)` the spatial padding size, and :math:`(d_Y, d_X)`
    the dilation factor of filter application. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H - (k_H - 1) * (d_Y - 1)) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W - (k_W - 1) * (d_X - 1)) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    """
    return convolution_2d.convolution_2d(x, W, b,
                                         stride, pad, cover_all, dilate=dilate)
