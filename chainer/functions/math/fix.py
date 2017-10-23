from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Fix(function_node.FunctionNode):

    @property
    def label(self):
        return 'fix'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.fix(x[0]), x[0].dtype),

    def backward(self, indexes, grad_outputs):
        return grad_outputs[0] * 0,


def fix(x):
    """Elementwise fix function.

    .. math::
       y_i = \\lfix x_i \\rfix

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """

    return Fix().apply((x,))[0]
