import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


def _get_extended_shape(beta, x):
    return (1,) + beta.shape + (1,) * (x.ndim - beta.ndim - 1)


def _get_reduction_axes(beta, x):
    return (0,) + tuple(range(1 + beta.ndim, x.ndim))


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


_preamble = '''
template <typename T> __device__ T sigmoid(T x) {
    const T half = 0.5;
    return tanh(x * half) * half + half;
}
'''


class Swish(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 'beta'))
        x_type, beta_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            beta_type.dtype == x_type.dtype,
            beta_type.ndim <= x_type.ndim - 1,
            beta_type.shape == x_type.shape[1:1 +
                                            type_check.eval(beta_type.ndim)]
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, beta = inputs
        beta = beta.reshape(_get_extended_shape(beta, x))

        y = x * _sigmoid(beta * x)
        return y,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, beta = inputs
        beta = beta.reshape(_get_extended_shape(beta, x))

        # Eliminating intermediate variable `bx` somehow degrades the
        # precision.
        y = cuda.elementwise(
            'T x, T beta', 'T y',
            '''
            T bx = beta * x;
            y = x * sigmoid(bx);
            ''',
            'swish_fwd', preamble=_preamble
        )(x, beta)
        return y,

    def backward(self, indexes, grad_outputs):
        x, beta = self.get_retained_inputs()
        gy, = grad_outputs

        shape = _get_extended_shape(beta, x)
        reduction_axes = _get_reduction_axes(beta, x)

        return SwishGrad(shape, reduction_axes).apply((x, beta, gy))


class SwishGrad(function_node.FunctionNode):

    def __init__(self, extended_shape, reduction_axes):
        super(SwishGrad, self).__init__()
        self.extended_shape = extended_shape
        self.reduction_axes = reduction_axes

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, beta, gy = inputs
        beta = beta.reshape(self.extended_shape)

        sig = _sigmoid(beta * x)
        y = x * sig
        by = beta * y
        one = x.dtype.type(1)

        gx = gy * (by + sig * (one - by))
        gb = gy * y * (x - y)
        gb = utils.force_array(gb.sum(axis=self.reduction_axes))
        return gx, gb

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, beta, gy = inputs
        beta = beta.reshape(self.extended_shape)

        gx, gb = cuda.elementwise(
            'T x, T beta, T gy', 'T gx, T gb',
            '''
            T bx = beta * x;
            T sig = sigmoid(bx);
            T y = x * sig;
            T by = beta * y;
            gx = gy * (by + sig * (1 - by));
            gb = gy * y * (x - y);
            ''',
            'swish_bwd', preamble=_preamble
        )(x, beta, gy)
        gb = utils.force_array(gb.sum(axis=self.reduction_axes))
        return gx, gb

    def backward(self, indexes, grad_outputs):
        x, beta, gy = self.get_retained_inputs()
        beta = chainer.functions.broadcast_to(
            beta.reshape(self.extended_shape), gy.shape)
        ggx, ggb = grad_outputs
        ggb = chainer.functions.broadcast_to(
            ggb.reshape(self.extended_shape), gy.shape)

        sig = chainer.functions.sigmoid(beta * x)
        y = x * sig
        by = beta * y
        one_minus_sig = 1 - sig
        sig_one_minus_by = sig * (1 - by)
        y_x_minus_y = y * (x - y)
        x_minus_2y = x - 2 * y

        ret = []

        common = gy * y * (2 + beta * x_minus_2y) * one_minus_sig
        if 0 in indexes:
            gx = ggx * gy * beta * one_minus_sig * \
                (by + 2 * sig_one_minus_by) + ggb * common
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            gb = ggx * common + ggb * gy * y_x_minus_y * x_minus_2y
            gb = chainer.functions.sum(gb, axis=self.reduction_axes)
            ret.append(chainer.functions.cast(gb, beta.dtype))
        if 2 in indexes:
            ggy = ggx * (by + sig_one_minus_by) + ggb * y_x_minus_y
            ret.append(chainer.functions.cast(ggy, gy.dtype))

        return ret


def swish(x, beta):
    """Swish activation function.

    .. math:: f(x, \\beta) = x \\cdot \\sigma(\\beta x),

    where :math:`\\sigma(\\cdot)` is the sigmoid function. It has the
    following properties:

    .. math::
        f(x, 0) &= \\frac{x}{2}, \\\\
        \\lim_{\\beta \\to \\infty} f(x, \\beta) &= \\max(0, x).

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable of
            shape :math:`(s_B, s_1, s_2, ..., s_N)`, where :math:`s_B` is
            assumed to be the *minibatch dimension*.
        beta (:class:`~chainer.Variable` or :ref:`ndarray`): Parameter variable
            :math:`\\beta` of shape :math:`(s_1, s_2, ..., s_M)`, where
            :math:`M` is an arbitrary integer between
            :math:`0 \\leq M \\leq N`. The number of dimensions of ``beta``
            will be matched with ``x`` by reshaping it as
            :math:`(1, s_1, ..., s_M, 1, ... 1)`, then ``beta`` and ``x``
            are multiplied together in an element-wise manner.

    Returns:
        ~chainer.Variable: Output variable of the same shape as ``x``.

    .. warning::
        :math:`\\beta` is a trainable parameter in the original paper
        (https://arxiv.org/abs/1710.05941). To train :math:`\\beta`, use
        :class:`chainer.links.Swish` instead.

    .. seealso::
        :class:`chainer.links.Swish` to manage the model parameter ``beta``.

    """
    y, = Swish().apply((x, beta))
    return y
