from __future__ import division

from chainer import backend
from chainer import function
from chainer.utils import type_check


class BinaryAccuracy(function.Function):

    ignore_label = -1

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 't'))
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            t_type.shape == x_type.shape,
        )

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        y, t = inputs
        # flatten
        y = y.ravel()
        t = t.ravel()
        c = (y >= 0)
        count = xp.maximum(1, (t != self.ignore_label).sum())
        return xp.asarray((c == t).sum() / count, dtype=y.dtype),


def binary_accuracy(y, t):
    """Computes binary classification accuracy of the minibatch.

    Args:
        y (:class:`~chainer.Variable` or :ref:`ndarray`):
            Array whose i-th element indicates the score of
            positive at the i-th sample.
            The prediction label :math:`\\hat t[i]` is ``1`` if
            ``y[i] >= 0``, otherwise ``0``.

        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            Array holding a signed integer vector of ground truth labels.
            If ``t[i] == 1``, it indicates that i-th sample is positive.
            If ``t[i] == 0``, it indicates that i-th sample is negative.
            If ``t[i] == -1``, corresponding ``y[i]`` is ignored.
            Accuracy is zero if all ground truth labels are ``-1``.

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    .. admonition:: Example

        We show the most common case, when ``y`` is the two dimensional array.

        >>> y = np.array([[-2.0, 0.0], # prediction labels are [0, 1]
        ...               [3.0, -5.0]]) # prediction labels are [1, 0]
        >>> t = np.array([[0, 1],
        ...              [1, 0]], np.int32)
        >>> F.binary_accuracy(y, t).array \
# 100% accuracy because all samples are correct.
        array(1.)
        >>> t = np.array([[0, 0],
        ...              [1, 1]], np.int32)
        >>> F.binary_accuracy(y, t).array \
# 50% accuracy because y[0][0] and y[1][0] are correct.
        array(0.5)
        >>> t = np.array([[0, -1],
        ...              [1, -1]], np.int32)
        >>> F.binary_accuracy(y, t).array \
# 100% accuracy because of ignoring y[0][1] and y[1][1].
        array(1.)
    """
    return BinaryAccuracy()(y, t)
