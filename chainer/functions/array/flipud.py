from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class FlipUD(function_node.FunctionNode):

    """Flip array in the up/down direction."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a',))
        a_type = in_types[0]

        type_check.expect(
            a_type.dtype.kind == 'f',
            a_type.ndim >= 1
        )

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        return xp.flipud(inputs[0]),

    def backward(self, indexes, grad_outputs):
        return FlipUD().apply(grad_outputs)


def flipud(a):
    """Flip array in the up/down direction.

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return FlipUD().apply((a,))[0]
