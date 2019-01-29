import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.math import sum as _sum
from chainer.utils import type_check


class BatchL2NormSquared(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0].reshape(len(inputs[0]), -1)
        return (x * x).sum(axis=1),

    def forward_gpu(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0].reshape(len(inputs[0]), -1)
        l2normsquared_kernel = cuda.reduce(
            'T x', 'T y', 'x * x', 'a + b', 'y = a', '0', 'l2normsquared'
        )
        return l2normsquared_kernel(x, axis=1),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()
        return BatchL2NormSquaredGrad().apply((x[0], gy[0]))


class BatchL2NormSquaredGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy0 = inputs
        gy0 = gy0.reshape(-1, *((1,) * (x.ndim - 1)))
        gx = 2 * x * gy0
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy0 = inputs
        gy0 = gy0.reshape(-1, *((1,) * (x.ndim - 1)))
        kernel = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = 2 * x * gy',
            'l2normsquared_bwd')
        gx = kernel(x, gy0)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy0 = self.get_retained_inputs()
        gy0 = gy0.reshape(-1, *((1,) * (x.ndim - 1)))
        gy0 = chainer.functions.broadcast_to(gy0, x.shape)
        ggx2 = 2 * grad_outputs[0]
        gx = ggx2 * gy0
        ggy0 = ggx2 * x
        return gx, _sum.sum(ggy0, axis=tuple(six.moves.range(1, ggy0.ndim)))


def batch_l2_norm_squared(x):
    """L2 norm (a.k.a.\\  Euclidean norm) squared.

    This function implements the square of L2 norm on a vector. No reduction
    along batch axis is done.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
            The first dimension is assumed to be the *minibatch dimension*.
            If ``x`` has more than two dimensions all but the first dimension
            are flattened to one dimension.

    Returns:
        ~chainer.Variable: Two dimensional output variable.

    """
    return BatchL2NormSquared().apply((x,))[0]
