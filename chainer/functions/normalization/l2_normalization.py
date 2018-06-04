import numpy

from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class NormalizeL2(function_node.FunctionNode):

    """L2 normalization"""

    def __init__(self, eps=1e-5, axis=1):
        self.eps = eps
        if isinstance(axis, int):
            axis = axis,
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = cuda.get_array_module(x)
        norm = (xp.sqrt(xp.sum(xp.square(x), axis=self.axis, keepdims=True))
                + x.dtype.type(self.eps))
        return utils.force_array(x / norm),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        F = chainer.functions

        norm_noeps = F.sqrt(F.sum(F.square(x), axis=self.axis, keepdims=True))
        norm = norm_noeps + self.eps
        norm = F.broadcast_to(norm, gy.shape)

        x_gy_reduced = F.sum((x * gy), axis=self.axis, keepdims=True)
        x_gy_reduced /= norm_noeps
        x_gy_reduced = F.broadcast_to(x_gy_reduced, gy.shape)
        gx = gy * norm - x_gy_reduced * x
        gx = gx / norm ** 2

        return gx,


def normalize(x, eps=1e-5, axis=1):
    """L2 norm squared (a.k.a.\\  Euclidean norm).

    This function implements L2 normalization on a vector along the given axis.
    No reduction is done along the normalization axis.

    In the case when :obj:`axis=1` and :math:`\\mathbf{x}` is a matrix of
    dimension :math:`(N, K)`, where :math:`N` and :math:`K` denote mini-batch
    size and the dimension of the input vectors, this function computes an
    output matrix :math:`\\mathbf{y}` of dimension :math:`(N, K)` by the
    following equation:

    .. math::
       \\mathbf{y}_i =
           {\\mathbf{x}_i \\over \\| \\mathbf{x}_i \\|_2 + \\epsilon}

    :obj:`eps` is used to avoid division by zero when norm of
    :math:`\\mathbf{x}` along the given axis is zero.

    The default value of :obj:`axis` is determined for backward compatibility.

    Args:
        x (~chainer.Variable): Two dimensional output variable. The first
            dimension is assumed to be the mini-batch dimension.
        eps (float): Epsilon value for numerical stability.
        axis (int or tuple of ints): Axis along which to normalize.

    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    """
    return NormalizeL2(eps, axis).apply((x,))[0]
