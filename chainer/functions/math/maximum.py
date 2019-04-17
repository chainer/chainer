import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Maximum(function_node.FunctionNode):
    """Element-wise maximum of input variables."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x1', 'x2'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
        )
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward_cpu(self, inputs):
        # may broadcast
        self.retain_inputs((0, 1))
        x1, x2 = inputs
        y = numpy.maximum(x1, x2)
        return utils.force_array(y),

    def forward_gpu(self, inputs):
        # may broadcast
        self.retain_inputs((0, 1))
        x1, x2 = inputs
        return cuda.cupy.maximum(x1, x2),

    def backward(self, indexes, grad_outputs):
        x1, x2 = self.get_retained_inputs()
        return MaximumGrad(x1.data, x2.data).apply((grad_outputs[0],))


class MaximumGrad(function_node.FunctionNode):

    def __init__(self, x1, x2):
        self.cond = x1 >= x2
        self.x1_shape = x1.shape
        self.x2_shape = x2.shape

    def forward_cpu(self, inputs):
        gy, = inputs
        gx1 = utils.force_array(numpy.where(self.cond, gy, gy.dtype.type(0)))
        gx2 = utils.force_array(numpy.where(self.cond, gy.dtype.type(0), gy))
        return (
            utils.sum_to(gx1, self.x1_shape),
            utils.sum_to(gx2, self.x2_shape))

    def forward_gpu(self, inputs):
        gy, = inputs
        gx1, gx2 = cuda.elementwise(
            'S cond, T gy', 'T gx1, T gx2',
            '''
            gx1 = cond ? gy : (T)0.0;
            gx2 = cond ? (T)0.0 : gy;
            ''',
            'maximum_bwd1')(self.cond, gy)
        return (
            utils.sum_to(gx1, self.x1_shape),
            utils.sum_to(gx2, self.x2_shape))

    def backward(self, indexes, grad_outputs):
        return chainer.functions.where(
            utils.force_array(self.cond), grad_outputs[0], grad_outputs[1]),


def maximum(x1, x2):
    """Element-wise maximum of input variables.

    Args:
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

        x2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be compared.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x1 = np.arange(6).astype(np.float32)
        >>> x1
        array([0., 1., 2., 3., 4., 5.], dtype=float32)
        >>> x2 = np.array([5, 4, 3, 2, 1, 0]).astype(np.float32)
        >>> x2
        array([5., 4., 3., 2., 1., 0.], dtype=float32)
        >>> y = F.maximum(x1, x2)
        >>> y.shape
        (6,)
        >>> y.array
        array([5., 4., 3., 3., 4., 5.], dtype=float32)

    """
    return Maximum().apply((x1, x2))[0]
