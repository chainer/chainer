from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Ceil(function_node.FunctionNode):

    @property
    def label(self):
        return 'ceil'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        self._xp = cuda.get_array_module(*x)
        return utils.force_array(self._xp.ceil(x[0]), x[0].dtype),

    def backward(self, indexes, grad_outputs):
        return grad_outputs[0] * 0,


def ceil(x):
    """Elementwise ceil function.

    .. math::
       y_i = \\lceil x_i \\rceil

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """

    return Ceil().apply((x,))[0]
