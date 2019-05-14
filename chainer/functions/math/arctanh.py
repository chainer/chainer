from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Arctanh(function_node.FunctionNode):

    """Elementwise inverse hyperbolic tangent function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = cuda.get_array_module(x)
        y = xp.arctanh(x)
        return utils.force_array(y, dtype=x.dtype),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        gx = 1. / (1 - x ** 2) * gy
        return gx,


def arctanh(x):
    """Elementwise inverse hyperbolic tangent function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Arctanh().apply((x,))[0]
