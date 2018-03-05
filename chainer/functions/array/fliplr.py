from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class FlipLR(function_node.FunctionNode):

    """Flip array in the left/right direction."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.fliplr(inputs[0]),

    def backward(self, indexes, grad_outputs):
        return FlipLR().apply(grad_outputs)


def fliplr(a):
    """Flip array in the left/right direction.

    Args:
        xs (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return FlipLR().apply((a,))[0]
