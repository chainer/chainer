import warnings

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.functions.array import reshape
from chainer.functions.array import tile
from chainer.functions.normalization import batch_normalization
from chainer.utils import argument
from chainer.utils import type_check
import chainerx


def _check_type_and_shape(x, gamma, beta, mean=None, var=None):
    if x.ndim <= 2:
        raise ValueError('Input dimension must be greater than 2, '
                         'including batch size dimension '
                         '(first dimension).')
    n_channels = x.shape[1]
    if x.dtype != gamma.dtype:
        raise type_check.InvalidType(x.dtype, gamma.dtype)
    if x.dtype != beta.dtype:
        raise type_check.InvalidType(x.dtype, beta.dtype)
    if n_channels != gamma.size:
        raise type_check.InvalidType(n_channels, gamma.size)
    if n_channels != beta.size:
        raise type_check.InvalidType(n_channels, beta.size)
    if mean is not None:
        if x.dtype != mean.dtype:
            raise type_check.InvalidType(x.dtype, mean.dtype)
        if n_channels != mean.size:
            raise type_check.InvalidType(n_channels, mean.size)
    if var is not None:
        if x.dtype != var.dtype:
            raise type_check.InvalidType(x.dtype, var.dtype)
        if n_channels != var.size:
            raise type_check.InvalidType(n_channels, var.size)


def _to_current_device(x):
    xp = backend.get_array_module(x)
    if xp is numpy:
        return x
    else:
        current_device_id = cuda.cupy.cuda.device.get_device_id()
        x_device_id = x.device.id
        if current_device_id != x_device_id:
            x = cuda.to_gpu(x, current_device_id)
        return x


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
    _check_type_and_shape(x, gamma, beta, running_mean, running_var)
    batch_size, channels = x.shape[:2]
    original_shape = x.shape
    x = reshape.reshape(x, (1, batch_size * channels) + original_shape[2:])
    gamma = tile.tile(gamma, batch_size)
    beta = tile.tile(beta, batch_size)

    xp = backend.get_array_module(x, gamma, beta)
    if running_mean is not None:
        if isinstance(running_mean, chainer.variable.Variable):
            tiled_mean = tile.tile(running_mean, batch_size)
        else:
            # running_mean = _to_current_device(running_mean)
            tiled_mean = xp.concatenate([running_mean] * batch_size)
    else:
        tiled_mean = None
    if running_var is not None:
        if isinstance(running_var, chainer.variable.Variable):
            tiled_var = tile.tile(running_var, batch_size)
        else:
            # running_var = _to_current_device(running_var)
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
            running_mean, tiled_mean = backend.from_chainerx(
                (running_mean, tiled_mean)
            )
            tiled_mean.reshape((batch_size, channels)).mean(axis=0)
        running_mean[:] = xp.reshape(
            tiled_mean, (batch_size, channels)).mean(axis=0)
        if xp is chainerx:
            running_mean = backend.to_chainerx(running_mean)
    if running_var is not None:
        if xp is chainerx:
            running_var, tiled_var = backend.from_chainerx(
                (running_var, tiled_var)
            )
            tiled_var.reshape((batch_size, channels)).mean(axis=0)
        running_var[:] = xp.reshape(
            tiled_var, (batch_size, channels)).mean(axis=0)
        if xp is chainerx:
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
    _check_type_and_shape(x, gamma, beta, mean, var)
    original_shape = x.shape
    batch_size, channels = original_shape[:2]
    gamma = tile.tile(gamma, batch_size)
    beta = tile.tile(beta, batch_size)
    if isinstance(mean, chainer.variable.Variable):
        tiled_mean = tile.tile(mean, batch_size)
        tiled_var = tile.tile(var, batch_size)
    else:
        xp = backend.get_array_module(mean, var)
        mean = _to_current_device(mean)
        tiled_mean = xp.concatenate([mean] * batch_size)
        var = _to_current_device(var)
        tiled_var = xp.concatenate([var] * batch_size)

    x = reshape.reshape(x, (1, batch_size * channels) + original_shape[2:])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = batch_normalization.fixed_batch_normalization(
            x, gamma, beta, tiled_mean, tiled_var, eps=eps,
        )

    y = reshape.reshape(y, original_shape)
    return y
