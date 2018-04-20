import six
import warnings

from chainer.backends import cuda
from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.normalization import batch_normalization


def group_normalization(x, n_groups, gamma, beta, eps=1e-5):
    """Group normalization function.

    This function implements a "group normalization"
    which divides the channels into groups and computes within each group
    the mean and variance, then normalize by these statistics,
    scales and shifts them.


    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Batch tensors.
            First dimension of this value must be the size of minibatch and
            second dimension must be the number of channels.
            Moreover, this value must have one or more following dimensions,
            such as height and width.
        n_groups (int):
            The number of channel groups.
            This value must be a divisor of the number of channels.
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        eps (float): Epsilon value for numerical stability of normalization.


    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    See: `Group Normalization <https://arxiv.org/abs/1803.08494>`_
    """
    if x.ndim <= 2:
        raise ValueError('Input dimension must be bigger than 2, '
                         'excluding batch size dimension '
                         '(first dimension).')

    if not isinstance(n_groups, int):
        raise TypeError('Argument: \'n_groups\' type must be (int).')

    xp = cuda.get_array_module(x)

    batch_size, channels = x.shape[:2]
    original_shape = x.shape

    if channels % n_groups != 0:
        raise ValueError('Argument: \'n_groups\' must be a divisor '
                         'of the number of channel.')

    # By doing this reshaping, calling batch_normalization function becomes
    # equivalent to Group Normalization.
    x = reshape.reshape(x, (1, batch_size * n_groups, -1))
    dummy_gamma = xp.ones(batch_size * n_groups).astype(xp.float32)
    dummy_beta = xp.zeros(batch_size * n_groups).astype(xp.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = batch_normalization.batch_normalization(
            x, dummy_gamma, dummy_beta, eps=eps)

    x = reshape.reshape(x, original_shape)

    target_shape = [1, channels] + [1 for _ in six.moves.xrange(x.ndim - 2)]
    gamma_broadcast = broadcast.broadcast_to(
        reshape.reshape(gamma, target_shape), x.shape)
    beta_broadcast = broadcast.broadcast_to(
        reshape.reshape(beta, target_shape), x.shape)

    return x * gamma_broadcast + beta_broadcast
