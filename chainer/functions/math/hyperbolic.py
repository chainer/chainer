from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Cosh(function_node.FunctionNode):

    @property
    def label(self):
        return 'cosh'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.cosh(x[0])),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()
        gx = sinh(x[0])
        gx *= gy[0]
        return gx,


def cosh(x):
    """Elementwise hyperbolic cosine function.

    .. math::
       y_i = \\cosh x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cosh().apply((x,))[0]


class Sinh(function_node.FunctionNode):

    @property
    def label(self):
        return 'sinh'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.sinh(x[0])),

    def backward(self, x, gy):
        x = self.get_retained_inputs()
        gx = cosh(x[0])
        gx *= gy[0]
        return gx,


def sinh(x):
    """Elementwise hyperbolic sine function.

    .. math::
       y_i = \\sinh x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sinh().apply((x,))[0]
