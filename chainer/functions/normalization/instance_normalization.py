import warnings

from chainer.functions.array import reshape
from chainer.functions.array import tile
from chainer.functions.normalization import batch_normalization


def instance_normalization(x, gamma, beta, eps=2e-5):
    """Instance normalization function.

    This function implements instance normalization
    which normalizes each sample by its mean and standard deviation.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Batch tensors.
            First dimension of this value must be the size of minibatch and
            second dimension must be the number of channels.
            Moreover, this value must have one or more following dimensions,
            such as height and width.
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        eps (float): Epsilon value for numerical stability of normalization.

    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization
           <https://arxiv.org/abs/1607.08022>`_

    """
    if x.ndim <= 2:
        raise ValueError('Input dimension must be greater than 2, '
                         'including batch size dimension '
                         '(first dimension).')

    batch_size, channels = x.shape[:2]
    original_shape = x.shape
    x = reshape.reshape(x, (1, batch_size * channels) + x.shape[2:])
    gamma = tile.tile(gamma, batch_size)
    beta = tile.tile(beta, batch_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = batch_normalization.batch_normalization(x, gamma, beta, eps=eps)

    x = reshape.reshape(x, original_shape)
    return x
