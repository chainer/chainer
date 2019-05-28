import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


def _fwd_kern():
    return cuda.elementwise(
        'T x, T cond, T W', 'T y',
        'y = cond >= 0 ? x : (T)(x * W)', 'prelu')


def _get_extended_shape(W, x):
    return (1,) + W.shape + (1,) * (x.ndim - W.ndim - 1)


def _get_reduce_axes(W, x):
    return (0,) + tuple(range(1 + W.ndim, x.ndim))


class PReLUFunction(function_node.FunctionNode):

    """Parametric Rectified Linear Unit function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 'W'))
        x_type, W_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            W_type.dtype == x_type.dtype,
            x_type.ndim >= W_type.ndim + 1,
            x_type.shape[1:1 + type_check.eval(W_type.ndim)] == W_type.shape
        )

    def forward_cpu(self, inputs):
        x, W = inputs
        y = x.copy()
        masked = numpy.ma.masked_greater_equal(y, 0, copy=False)
        shape = _get_extended_shape(W, y)
        masked *= W.reshape(shape)
        self.retain_inputs((0, 1))
        return y,

    def forward_gpu(self, inputs):
        x, W = inputs
        shape = _get_extended_shape(W, x)
        y = _fwd_kern()(x, x, W.reshape(shape))
        self.retain_inputs((0, 1))
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs
        return PReLUFunctionGrad(
            x.data, _get_reduce_axes(W, x),
            _get_extended_shape(W, x)).apply((x, W, gy))


class PReLUFunctionGrad(function_node.FunctionNode):

    """Parametric Rectified Linear Unit gradient function."""

    def __init__(self, cond, reduce_axes, extended_shape):
        self.cond = cond
        self.reduce_axes = reduce_axes
        self.extended_shape = extended_shape

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 'W', 'gy'))
        x_type, W_type, gy_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            W_type.dtype == x_type.dtype,
            gy_type.dtype == x_type.dtype,
            x_type.ndim >= W_type.ndim + 1,
            x_type.shape[1:1 + type_check.eval(W_type.ndim)] == W_type.shape,
            gy_type.shape == x_type.shape
        )

    def forward_cpu(self, inputs):
        x, W, gy = inputs
        mask = self.cond >= 0
        masked = numpy.where(mask, 0, x * gy)

        if self.reduce_axes is None:
            # Reached from backward() of PReLUFunctionGrad i.e. this class, to
            # compute higher order derivatives
            gW = masked
        else:
            # Reached from backward() of PReLUFunction, to compute first
            # derivatives
            gW = masked.sum(axis=self.reduce_axes)

        if numpy.isscalar(gW):
            gW = numpy.array(gW)

        gx = gy.copy()
        masked = numpy.ma.array(gx, mask=mask)
        masked *= W.reshape(self.extended_shape)
        self.retain_inputs((0, 1, 2))
        return gx, gW

    def forward_gpu(self, inputs):
        x, W, gy = inputs
        masked = cuda.elementwise(
            'T x, T cond, T gy', 'T masked',
            'masked = cond >= 0 ? (T)0 : (T)(x * gy)',
            'prelu_masked')(x, self.cond, gy)

        if self.reduce_axes is None:
            gW = masked.copy()
        else:
            gW = masked.sum(axis=self.reduce_axes)

        gx = masked  # reuse buffer
        _fwd_kern()(gy, self.cond, W.reshape(self.extended_shape), gx)
        self.retain_inputs((0, 1, 2))
        return gx, gW

    def backward(self, indexes, grad_outputs):
        x, W, gy = self.get_retained_inputs()
        ggx, ggW = grad_outputs

        ggW = chainer.functions.broadcast_to(
            chainer.functions.reshape(ggW, self.extended_shape), x.shape)
        ggW *= self.cond < 0

        gxgy, gxW = (
            PReLUFunctionGrad(self.cond, None, self.extended_shape)
            .apply((gy, W, ggx))
        )

        ret = []
        if 0 in indexes:
            ret.append(gy * ggW)
        if 1 in indexes:
            ret.append(chainer.functions.sum(gxW, axis=self.reduce_axes))
        if 2 in indexes:
            ret.append(x * ggW + gxgy)
        return ret


def prelu(x, W):
    """Parametric ReLU function.

    It accepts two arguments: an input ``x`` and a weight array ``W``
    and computes the output as

    .. math::
        PReLU(x_i) = \\begin{cases}
        x_i & (x_i>0) \\\\ W_i * x_i & (otherwise)\\end{cases}

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
            Its first axis is assumed to be the minibatch dimension.
        W (:class:`~chainer.Variable` or :ref:`ndarray`): Weight variable.

    Returns:
        ~chainer.Variable: Output variable

    .. admonition:: Example

        >>> x = np.arange(-3, 3, dtype=np.float32).reshape((2, 3))
        >>> x
        array([[-3., -2., -1.],
               [ 0.,  1.,  2.]], dtype=float32)
        >>> W = np.array([0.01, 0.1, 1], dtype=np.float32)
        >>> W
        array([0.01, 0.1 , 1.  ], dtype=float32)
        >>> F.prelu(x, W)
        variable([[-0.03, -0.2 , -1.  ],
                  [ 0.  ,  1.  ,  2.  ]])

    .. note::
        When the PReLU function is combined with two-dimensional convolution,
        the elements of parameter :math:`W` are typically shared across the
        same filter of different pixels. In order to support such usage,
        this function supports the shape of parameter array that indicates
        leading dimensions of input arrays except the batch dimension.

        For example, if :math:`W` has the shape of :math:`(2, 3, 4)`,
        :math:`x` must have the shape of :math:`(B, 2, 3, 4, S_1, ..., S_N)`
        where :math:`B` is the batch size and the number of trailing
        :math:`S`'s :math:`N` is an arbitrary non-negative integer.

    .. seealso:: :class:`chainer.links.PReLU`

    """
    return PReLUFunction().apply((x, W))[0]
