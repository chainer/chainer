from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class FlipUD(function_node.FunctionNode):

    """Flip array in the up/down direction."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 1
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.flipud(inputs[0]),

    def backward(self, indexes, grad_outputs):
        return FlipUD().apply(grad_outputs)


def flipud(a):
    """Flip array in the up/down direction.

    Args:
        xs (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return FlipUD().apply((a,))[0]
