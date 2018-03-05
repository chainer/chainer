from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class Fmod(function_node.FunctionNode):

    @property
    def label(self):
        return 'fmod'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types.size() == 2,
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].dtype.kind == 'f',
            in_types[1].dtype.kind == 'f',
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = cuda.get_array_module(*inputs)
        x, divisor = inputs
        m = xp.fmod(x, divisor)
        return utils.force_array(m, x.dtype),

    def backward(self, indexes, grad_outputs):
        x, divisor = self.get_retained_inputs()
        gw, = grad_outputs
        return gw, - chainer.functions.fix(x / divisor) * gw


def fmod(x, divisor):
    """Elementwise mod function.

    .. math::
       y_i = x_i \\bmod \\mathrm{divisor}.

    Args:
        x (~chainer.Variable): Input variable.
        divisor (~chainer.Variable): Input divisor.
    Returns:
        ~chainer.Variable: Output variable.
    """
    return Fmod().apply((x, divisor))[0]
