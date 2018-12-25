import warnings

from chainer import backend
from chainer.functions.array import reshape
from chainer.functions.array import tile
from chainer.functions.normalization import batch_normalization
from chainer.utils import argument
from chainer.utils import type_check
import chainerx


def instance_normalization(x, gamma, beta, **kwargs):
    """Instance normalization function.

    This function implements instance normalization
    which normalizes each sample by its mean and standard deviation.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Batch tensors.
            First dimension of this value must be the size of minibatch and
            second dimension must be the number of channels.
            Moreover, this value must have one or more following dimensions,
            such as height and width.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`):
            Scaling parameter of normalized data with the shape of (C,).
        beta (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of normalized data with the shape o f(C,).
        running_mean (:class:`~chainer.Variable`, :ref:`ndarray`, or None):
            Shifting parameter of input with the shape of (C,).
        running_var (:class:`~chainer.Variable`, :ref:`ndarray`, or None):
            Scaling parameter of input with the shape of (C,).
        decay (float): Decay rate of moving average. It is used during
            training.

    Returns:
        :class:`~chainer.Variable`: The output variable which has the same
        shape as :math:`x`.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization
           <https://arxiv.org/abs/1607.08022>`_

    """
    eps, running_mean, running_var, decay = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9)
    )
    if x.ndim <= 2:
        raise ValueError('Input dimension must be greater than 2, '
                         'including batch size dimension '
                         '(first dimension).')
    batch_size, channels = x.shape[:2]
    original_shape = x.shape
    assert channels == gamma.size and channels == beta.size
    if channels != gamma.size:
        raise ValueError
    if channels != beta.size:
        raise ValueError
    # type_check.expect(
    #     channels == gamma.size,
    #     channels == beta.size,
    # )
    x = reshape.reshape(x, (1, batch_size * channels) + original_shape[2:])
    gamma = tile.tile(gamma, batch_size)
    beta = tile.tile(beta, batch_size)

    xp = backend.get_array_module(x)
    if running_mean is not None:
        tiled_mean = xp.concatenate([running_mean] * batch_size)
    else:
        tiled_mean = None
    if running_var is not None:
        tiled_var = xp.concatenate([running_var] * batch_size)
    else:
        tiled_var = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = batch_normalization.batch_normalization(
            x, gamma, beta, eps=eps,
            running_mean=tiled_mean, running_var=tiled_var
        )

    y = reshape.reshape(y, original_shape)
    if running_mean is not None:
        if xp is chainerx:
            running_mean, running_var = backend.from_chainerx(
                (running_mean, running_var)
            )
            tiled_mean, tiled_var = backend.from_chainerx(
                (tiled_mean, tiled_var)
            )
        running_mean[:] = xp.reshape(
            tiled_mean, (batch_size, channels)).mean(axis=0)
        running_var[:] = xp.reshape(
            tiled_var, (batch_size, channels)).mean(axis=0)
        if xp is chainerx:
            running_mean = backend.to_chainerx(running_mean)
            running_var = backend.to_chainerx(running_var)
    return y


def fixed_instance_normalization(x, gamma, beta, mean, var, eps=2e-5):
    """Instance Normalization with fixed statistics.

    This is a variant of instance normalization, where the mean and variance
    are given by the caller as fixed variables. This is used on testing mode
    of the instance normalization layer with
    ``track_running_stats`` of ``True``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`):
            Scaling parameter of normalized data with the shape of (C,).
        beta (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of normalized data with the shape o f(C,).
        mean (:class:`~chainer.Variable` or :ref:`ndarray`):
            Shifting parameter of input with the shape of (C,).
        var (:class:`~chainer.Variable` or :ref:`ndarray`):
            Scaling parameter of input with the shape of (C,).
        eps (float): Epsilon value for numeircal stability.

    .. seealso::
       :func:`~chainer.functions.instance_normalization`,
       :class:`~chainer.links.InstancNormalization`

    """
    if x.ndim <= 2:
        raise ValueError('Input dimension must be greater than 2, '
                         'including batch size dimension '
                         '(first dimension).')
    batch_size, channels = x.shape[:2]
    original_shape = x.shape
    if channels != gamma.size:
        raise ValueError
    if channels != beta.size:
        raise ValueError
    if channels != mean.size:
        raise ValueError
    if channels != var.size:
        raise ValueError
    x = reshape.reshape(x, (1, batch_size * channels) + original_shape[2:])
    gamma = tile.tile(gamma, batch_size)
    beta = tile.tile(beta, batch_size)
    xp = backend.get_array_module(x)
    tiled_mean = xp.concatenate([mean] * batch_size)
    tiled_var = xp.concatenate([var] * batch_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = batch_normalization.fixed_batch_normalization(
            x, gamma, beta, tiled_mean, tiled_var, eps=eps,
        )

    x = reshape.reshape(x, original_shape)
    return x
