from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx


class AbsoluteError(function_node.FunctionNode):

    """Element-wise absolute error function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_chainerx(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        return chainerx.abs(self.diff),

    def forward(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        return utils.force_array(abs(self.diff), dtype=x0.dtype),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        gx = gy * backend.get_array_module(gy).sign(self.diff)
        return gx, -gx


def absolute_error(x0, x1):
    """Element-wise absolute error function.

    Computes the element-wise absolute error :math:`L` between two inputs
    :math:`x_0` and :math:`x_1` defined as follows.

    .. math::

        L = |x_0 - x_1|

    Args:
        x0 (:class:`~chainer.Variable` or :ref:`ndarray`):
            First input variable.
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Second input variable.

    Returns:
        ~chainer.Variable:
            An array representing the element-wise absolute error between the
            two inputs.

    """
    return AbsoluteError().apply((x0, x1))[0]
