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
        x2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.

    Returns:
        ~chainer.Variable: ``(x1 - x2) ** 2`` element-wise.
    """
    return SquaredDifference().apply((x1, x2))[0]
