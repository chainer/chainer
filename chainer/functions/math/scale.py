import chainer
from chainer.functions.array import broadcast
from chainer.functions.array import reshape


def scale(x, y, axis=1):
    """Elementwise product with broadcasting.

    Computes a elementwise product of two input variables, with the shape of
    the latter variable broadcasted to match the shape of the former. ``axis``
    is the first axis of the first variable along which the second variable is
    applied.

    The term "broadcasting" here comes from Caffe's scale layer so the
    "broadcasting" with the following arguments::

           x : 100 x 3 x 40 x 5 x 6
           y : 3 x 40
        axis : 1

    is equivalent to the following numpy broadcasting::

        x : 100 x  3 x 40 x 5 x 6
        y :  (1 x) 3 x 40 x 1 x 1

    Note that the axis of ``x`` to which we apply ``y`` is specified by the
    argument ``axis``, whose meaning is different from numpy's ``axis``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable to be scaled.
        y (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable to scale, broadcasted.
        axis (int): The first axis of ``x`` along which ``y`` is applied.

    Returns:
        ~chainer.Variable: Output variable.

    """
    x_shape = x.shape
    y_shape = y.shape
    if chainer.is_debug():
        assert x_shape[axis:axis + len(y_shape)] == y_shape
    y1_shape = tuple([1] * axis + list(y_shape) +
                     [1] * (len(x_shape) - axis - len(y_shape)))
    y1 = reshape.reshape(y, y1_shape)
    y2 = broadcast.broadcast_to(y1, x_shape)
    return x * y2
