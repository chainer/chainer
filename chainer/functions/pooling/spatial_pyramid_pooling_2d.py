import math
import six
import warnings

import chainer


def spatial_pyramid_pooling_2d(x, pyramid_height, pooling_class=None,
                               pooling=None):
    """Spatial pyramid pooling function.

    It outputs a fixed-length vector regardless of input feature map size.

    It performs pooling operation to the input 4D-array ``x`` with different
    kernel sizes and padding sizes, and then flattens all dimensions except
    first dimension of all pooling results, and finally concatenates them along
    second dimension.

    At :math:`i`-th pyramid level, the kernel size
    :math:`(k_h^{(i)}, k_w^{(i)})` and padding size
    :math:`(p_h^{(i)}, p_w^{(i)})` of pooling operation are calculated as
    below:

    .. math::
        k_h^{(i)} &= \\lceil b_h / 2^i \\rceil, \\\\
        k_w^{(i)} &= \\lceil b_w / 2^i \\rceil, \\\\
        p_h^{(i)} &= (2^i k_h^{(i)} - b_h) / 2, \\\\
        p_w^{(i)} &= (2^i k_w^{(i)} - b_w) / 2,

    where :math:`\\lceil \\cdot \\rceil` denotes the ceiling function, and
    :math:`b_h, b_w` are height and width of input variable ``x``,
    respectively. Note that index of pyramid level :math:`i` is zero-based.

    See detail in paper: `Spatial Pyramid Pooling in Deep Convolutional \
    Networks for Visual Recognition \
    <https://arxiv.org/abs/1406.4729>`_.

    Args:
        x (~chainer.Variable): Input variable. The shape of ``x`` should be
            ``(batchsize, # of channels, height, width)``.
        pyramid_height (int): Number of pyramid levels
        pooling_class (MaxPooling2D):
            *(deprecated since v4.0.0)* Only MaxPooling2D is supported.
            Please use the ``pooling`` argument instead since this argument is
            deprecated.
        pooling (str):
            Currently, only ``max`` is supported, which performs a 2d max
            pooling operation. Replaces the ``pooling_class`` argument.

    Returns:
        ~chainer.Variable: Output variable. The shape of the output variable
        will be :math:`(batchsize, c \\sum_{h=0}^{H-1} 2^{2h}, 1, 1)`,
        where :math:`c` is the number of channels of input variable ``x``
        and :math:`H` is the number of pyramid levels.

    .. note::

        This function uses some pooling classes as components to perform
        spatial pyramid pooling. Currently, it only supports
        :class:`~functions.MaxPooling2D` as elemental pooling operator so far.

    """

    bottom_c, bottom_h, bottom_w = x.shape[1:]
    ys = []

    # create pooling functions for different pyramid levels and apply it
    for pyramid_level in six.moves.range(pyramid_height):
        num_bins = int(2 ** pyramid_level)

        ksize_h = int(math.ceil(bottom_h / (float(num_bins))))
        remainder_h = ksize_h * num_bins - bottom_h
        pad_h = remainder_h // 2

        ksize_w = int(math.ceil(bottom_w / (float(num_bins))))
        remainder_w = ksize_w * num_bins - bottom_w
        pad_w = remainder_w // 2

        ksize = (ksize_h, ksize_w)
        pad = (pad_h, pad_w)

        if pooling_class is not None:
            warnings.warn('pooling_class argument is deprecated. Please use '
                          'the pooling argument.', DeprecationWarning)

        if (pooling_class is None) == (pooling is None):
            raise ValueError('Specify the pooling operation either using the '
                             'pooling_class or the pooling argument.')

        if (pooling_class is chainer.functions.MaxPooling2D or
                pooling == 'max'):
            pooler = chainer.functions.MaxPooling2D(
                ksize=ksize, stride=None, pad=pad, cover_all=True)
        else:
            pooler = pooling if pooling is not None else pooling_class
            raise ValueError('Unsupported pooling operation: ', pooler)

        y_var = pooler.apply((x,))[0]
        n, c, h, w = y_var.shape
        ys.append(y_var.reshape((n, c * h * w, 1, 1)))

    return chainer.functions.concat(ys)
