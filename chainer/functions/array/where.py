import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Where(function_node.FunctionNode):

    """Choose elements depending on condition."""

    def __init__(self, condition):
        self.condition = condition

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 'y'))
        x_type, y_type = in_types
        condition = type_check._make_variable_from_array(
            # allow scalar `condition`
            chainer.utils.force_array(self.condition),
            'condition')

        type_check.expect(
            condition.dtype == numpy.bool_,
            x_type.dtype == y_type.dtype,
        )
        type_check.expect_broadcast_shapes(
            condition.shape, x_type.shape, y_type.shape)

    def forward(self, inputs):
        # may broadcast
        xp = backend.get_array_module(*inputs)
        x, y = inputs
        condition = self.condition
        return xp.where(condition, x, y),

    def backward(self, indexes, grad_outputs):
        condition = self.condition
        xp = backend.get_array_module(condition)
        g, = grad_outputs
        zero = xp.zeros((), dtype=g.dtype)
        ret = []
        if 0 in indexes:
            gx, = Where(condition).apply((g, zero))
            ret.append(chainer.functions.sum_to(gx, self.inputs[0].shape))
        if 1 in indexes:
            gy, = Where(condition).apply((zero, g))
            ret.append(chainer.functions.sum_to(gy, self.inputs[1].shape))
        return ret


def where(condition, x, y):
    """Choose elements depending on condition.

    This function choose values depending on a given ``condition``.
    All ``condition``, ``x``, and ``y`` must have the same shape.

    Args:
        condition (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable containing the condition.
            A :math:`(s_1, s_2, ..., s_N)` -shaped boolean array.
            Only boolean array is permitted.
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable chosen when ``condition`` is ``True``.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        y (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable chosen when ``condition`` is ``False``.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

    Returns:
        ~chainer.Variable: Variable containing chosen values.

    .. admonition:: Example

        >>> cond = np.array([[1, 0], [0, 1]], dtype=np.bool)
        >>> cond
        array([[ True, False],
               [False,  True]])
        >>> x = np.array([[1, 2], [3, 4]], np.float32)
        >>> y = np.zeros((2, 2), np.float32)
        >>> F.where(cond, x, y).array
        array([[1., 0.],
               [0., 4.]], dtype=float32)

    """
    if isinstance(condition, chainer.Variable):
        condition = condition.array

    z, = Where(condition).apply((x, y))
    return z
