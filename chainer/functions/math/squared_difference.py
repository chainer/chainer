from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class SquaredDifference(function_node.FunctionNode):
    """Squared difference of input variables."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x1', 'x2'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        x1, x2 = inputs
        difference = x1 - x2
        y = xp.square(difference)
        return utils.force_array(y, dtype=x1.dtype),

    def backward(self, indexes, grads):
        gy, = grads
        x1, x2 = self.get_retained_inputs()
        difference = x1 - x2
        gx = gy * 2 * difference
        return gx, -gx


def squared_difference(x1, x2):
    """Squared difference of input variables.

    Args:
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

        x2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

    Returns:
        ~chainer.Variable: ``(x1 - x2) ** 2`` element-wise.
        A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

    .. admonition:: Example

        >>> x1 = np.arange(6).astype(np.float32)
        >>> x1
        array([0., 1., 2., 3., 4., 5.], dtype=float32)
        >>> x2 = np.array([5, 4, 3, 2, 1, 0]).astype(np.float32)
        >>> x2
        array([5., 4., 3., 2., 1., 0.], dtype=float32)
        >>> y = F.squared_difference(x1, x2)
        >>> y.shape
        (6,)
        >>> y.array
        array([25.,  9.,  1.,  1.,  9., 25.], dtype=float32)

    """
    return SquaredDifference().apply((x1, x2))[0]
