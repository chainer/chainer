from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Floor(function_node.FunctionNode):

    @property
    def label(self):
        return 'floor'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, inputs):
        x = inputs[0]
        xp = cuda.get_array_module(x)
        return utils.force_array(xp.floor(x), x.dtype),

    def backward(self, indexes, grad_outputs):
        return grad_outputs[0] * 0,


def floor(x):
    """Elementwise floor function.

    .. math::
       y_i = \\lfloor x_i \\rfloor

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Floor().apply((x,))[0]
