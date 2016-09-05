from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Square(function.Function):
    @property
    def label(self):
        return 'square'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.square(x[0], dtype=x[0].dtype)),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(x[0]*2.0, dtype=x[0].dtype)
        gx *= gy[0]
        return gx,


def square(x):
    """Elementwise square function.

    .. math::
       y_i = x_i*x_i .

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Square()(x)
