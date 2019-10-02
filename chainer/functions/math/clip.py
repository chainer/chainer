import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Clip(function_node.FunctionNode):
    """Clips (limits) elements of input variable."""

    def __init__(self, x_min, x_max):
        if x_min is None and x_max is None:
            raise ValueError('must set either max or min')

        # x_min must be less than x_max.
        if (x_min is not None) and (x_max is not None) and (x_min >= x_max):
            raise ValueError('x_min must be less than x_max.')
        self.x_min = x_min
        self.x_max = x_max

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        return utils.force_array(
            numpy.clip(x, self.x_min, self.x_max),
            x.dtype),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.cupy.clip(x[0], self.x_min, self.x_max),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        return ClipGrad(x.data, self.x_min, self.x_max).apply(grad_outputs)


class ClipGrad(function_node.FunctionNode):

    def __init__(self, x, x_min, x_max):
        if x_min is None and x_max is None:
            raise ValueError('must set either max or min')

        self.cond = True
        if x_min is not None:
            self.cond *= (x_min <= x)
        if x_max is not None:
            self.cond *= (x <= x_max)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('gy',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        gy, = inputs
        gx = utils.force_array(gy * self.cond, gy.dtype)
        return gx,

    def forward_gpu(self, inputs):
        gx = cuda.elementwise(
            'T gy, bool cond', 'T gx',
            'gx = cond ? gy : T(0)',
            'clip_bwd')(inputs[0], self.cond)
        return gx,

    def backward(self, indexes, grad_outputs):
        return grad_outputs[0] * self.cond,


def clip(x, x_min, x_max):
    """Clips (limits) elements of input variable.

    Given an interval ``[x_min, xmax]``, elements outside the interval are
    clipped to the interval edges.

    Its gradients at ``x_min`` and ``x_max`` are regarded as 1.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable to be clipped.
        x_min (float): Minimum value.
        x_max (float): Maximum value.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Clip(x_min, x_max).apply((x,))[0]
