from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class Minimum(function_node.FunctionNode):
    """Element-wise minimum of input variables."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x1', 'x2'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
        )
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward(self, inputs):
        # may broadcast
        self.retain_inputs((0, 1))
        x1, x2 = inputs
        xp = backend.get_array_module(x1, x2)
        return utils.force_array(xp.minimum(x1, x2)),

    def backward(self, indexes, grad_outputs):
        x1, x2 = self.get_retained_inputs()
        return MinimumGrad(x1.data, x2.data).apply((grad_outputs[0],))


class MinimumGrad(function_node.FunctionNode):

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def forward_cpu(self, inputs):
        gy, = inputs
        x1, x2 = self.x1, self.x2
        gx1 = utils.force_array(gy * (x1 <= x2))
        gx2 = utils.force_array(gy * (x1 > x2))
        return utils.sum_to(gx1, x1.shape), utils.sum_to(gx2, x2.shape)

    def forward_gpu(self, inputs):
        gy, = inputs
        x1, x2 = self.x1, self.x2
        gx1 = cuda.elementwise(
            'T x1, T x2, T gy', 'T gx1',
            'gx1 = (x1 <= x2) ? gy : (T)0.0',
            'minimum_bwd1')(x1, x2, gy)
        gx2 = cuda.elementwise(
            'T x1, T x2, T gy', 'T gx1',
            'gx1 = (x1 > x2) ? gy : (T)0.0',
            'minimum_bwd2')(x1, x2, gy)
        return utils.sum_to(gx1, x1.shape), utils.sum_to(gx2, x2.shape)

    def backward(self, indexes, grad_outputs):
        x1, x2 = self.x1, self.x2
        cond = utils.force_array(x1 <= x2)
        ggy = chainer.functions.where(cond, grad_outputs[0], grad_outputs[1])
        return ggy,


def minimum(x1, x2):
    """Element-wise minimum of input variables.

    Args:
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.
        x2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Minimum().apply((x1, x2))[0]
