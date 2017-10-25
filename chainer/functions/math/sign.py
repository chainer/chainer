import numpy

from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Sign(function_node.FunctionNode):

    """Element-wise sign function."""

    @property
    def label(self):
        return 'sign'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward_cpu(self, x):
        return utils.force_array(numpy.sign(x[0])),

    def forward_gpu(self, x):
        return cuda.cupy.sign(x[0]),

    def backward(self, indexes, grad_outputs):
        return grad_outputs[0] * 0,


def sign(x):
    """Elementwise sign function.

    For a given input :math:`x`, it computed the following.

    .. math::

        sgn(x) = \\left \\{ \\begin{array}{cc}
        -1 & {\\rm if~x < 0} \\\\
        0 & {\\rm if~x = 0} \\\\
        1 & {\\rm if~x > 0} \\\\
        \\end{array} \\right.

    .. note::

        The gradient of this function is 0 everywhere.

    Args:
        x (~chainer.Variable): Input variable for which the sign is computed.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Sign().apply((x,))[0]
