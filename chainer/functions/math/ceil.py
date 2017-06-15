from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Ceil(function.Function):

    @property
    def label(self):
        return 'ceil'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self.retain_inputs(())
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.ceil(x[0]), x[0].dtype),

    def backward(self, x, grad_outputs):
        xp = cuda.get_array_module(*grad_outputs)
        return xp.zeros(self._in_shape, self._in_dtype),


def ceil(x):
    """Elementwise ceil function.

    .. math::
       y_i = \\lceil x_i \\rceil

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """

    return Ceil()(x)
