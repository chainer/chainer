from chainer import function_node
from chainer.utils import type_check


class Swapaxes(function_node.FunctionNode):
    """Swap two axes of an array."""

    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @property
    def label(self):
        return 'Swapaxes'

    def forward(self, inputs):
        x, = inputs
        return x.swapaxes(self.axis1, self.axis2),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        return Swapaxes(self.axis1, self.axis2).apply((gy,))


def swapaxes(x, axis1, axis2):
    """Swap two axes of a variable.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axis1 (int): The first axis to swap.
        axis2 (int): The second axis to swap.

    Returns:
        ~chainer.Variable: Variable whose axes are swapped.

    .. admonition:: Example

        >>> x = np.array([[[0, 1, 2], [3, 4, 5]]], np.float32)
        >>> x.shape
        (1, 2, 3)
        >>> y = F.swapaxes(x, axis1=0, axis2=1)
        >>> y.shape
        (2, 1, 3)
        >>> y.array
        array([[[0., 1., 2.]],
        <BLANKLINE>
               [[3., 4., 5.]]], dtype=float32)

    """
    y, = Swapaxes(axis1, axis2).apply((x,))
    return y
