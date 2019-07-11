import six

from chainer import backend
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class _SetItemZero(function_node.FunctionNode):

    """Write values to mask of zero-initialized array"""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        y = xp.zeros(self.mask.shape, x.dtype)
        y[self.mask] = x
        return y,

    def backward(self, indices, grad_outputs):
        g, = grad_outputs
        return g[self.mask],


class NormalizeL2(function_node.FunctionNode):

    """L2 normalization"""

    def __init__(self, eps=1e-5, axis=1):
        self.eps = eps
        if isinstance(axis, six.integer_types):
            axis = axis,
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = backend.get_array_module(x)
        # Note: Passing dtype argument to numpy.sqrt() because NumPy in
        # Python 2 looks to return a casted value to float32 when it takes a
        # float16 value.
        norm = (xp.sqrt(xp.sum(xp.square(x), axis=self.axis, keepdims=True),
                        dtype=x.dtype)
                + x.dtype.type(self.eps))
        return utils.force_array(x / norm),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        F = chainer.functions

        norm_noeps = F.sqrt(F.sum(F.square(x), axis=self.axis, keepdims=True))
        norm = norm_noeps + self.eps

        x_gy_reduced = F.sum((x * gy), axis=self.axis, keepdims=True)

        # L2 normalize with eps has continuous backward. However,
        # the backward is not differentiable for the indices of zero vectors.
        # To avoid nan in double backward, do not compute outside of mask.
        mask = norm_noeps.array != 0
        x_gy_reduced, = _SetItemZero(mask).apply((
            x_gy_reduced[mask] / norm_noeps[mask],))

        gx = gy * norm - x_gy_reduced * x
        gx = gx / norm ** 2

        return gx,


def normalize(x, eps=1e-5, axis=1):
    """Normalize input by L2 norm.

    This function implements L2 normalization on a sample along the given
    axis/axes. No reduction is done along the normalization axis.

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
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            multi-dimensional output variable. The first
            dimension is assumed to be the mini-batch dimension.
        eps (float): Epsilon value for numerical stability.
        axis (int or tuple of ints): Axis along which to normalize.

    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    """
    return NormalizeL2(eps, axis).apply((x,))[0]
