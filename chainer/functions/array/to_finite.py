import numpy

import chainer
from chainer import backend
from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.array import where


def _broadcast_with_axis(x, y, axis):
    x_shape = x.shape
    y_shape = y.shape
    if chainer.is_debug():
        assert x_shape[axis:axis + len(y_shape)] == y_shape
    y1_shape = tuple([1] * axis + list(y_shape) +
                     [1] * (len(x_shape) - axis - len(y_shape)))
    y1 = reshape.reshape(y, y1_shape)
    y2 = broadcast.broadcast_to(y1, x_shape)
    return y2


def to_finite(x, nan_x, posinf_x, neginf_x, axis=1):
    """Force an array that has NaN or infinite values to finite values.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            input variable.
        nan_x (:class:`~chainer.Variable` or :ref:`ndarray`):
            variable to replace NaN values in ``x``
        posinf_x (:class:`~chainer.Variable` or :ref:`ndarray`):
            variable to replace positive infinite values in ``x``
        neginf_x (:class:`~chainer.Variable` or :ref:`ndarray`):
            variable to replace negative infinite values in ``x``
        axis (int): The first axis of ``x`` along which ``nan_x``, ``posinf_x``
            and ``neginf_x`` is applied.

    Returns:
        ~chainer.Variable:
            An array forced to finite values.
    """
    nan_x = _broadcast_with_axis(x, nan_x, axis=axis)
    posinf_x = _broadcast_with_axis(x, posinf_x, axis=axis)
    neginf_x = _broadcast_with_axis(x, neginf_x, axis=axis)

    xp = backend.get_array_module(x)
    x = chainer.as_variable(x)
    x = where.where(xp.isnan(x.data), nan_x, x)
    if xp == numpy:
        x = where.where(xp.isposinf(x.data), posinf_x, x)
        x = where.where(xp.isneginf(x.data), neginf_x, x)
    else:
        x = where.where(xp.isinf(x.data) * (x.data > 0), posinf_x, x)
        x = where.where(xp.isinf(x.data) * (x.data < 0), neginf_x, x)
    return x
