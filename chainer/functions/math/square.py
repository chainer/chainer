from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Square(function_node.FunctionNode):

    @property
    def label(self):
        return 'square'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self.retain_inputs((0,))
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.square(x[0], dtype=x[0].dtype)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        gx = gy[0] * 2.0 * x
        return gx,


def square(x):
    """Elementwise square function.

    .. math::
       y_i = x_i ^ 2.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Square().apply((x,))[0]
