import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check
import chainerx

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _algorithm = cuda.cuda.cudnn.CUDNN_SOFTMAX_LOG  # type: ignore


def logsumexp(x, axis):
    xp = backend.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def _log_softmax(x, axis=1):
    if chainer.should_use_cudnn('>=auto'):
        xp = backend.get_array_module(x)
        if xp is cuda.cupy:
            return cudnn.softmax_forward(x, axis, _algorithm)
    log_z = logsumexp(x, axis)
    y = x - log_z
    return y


class LogSoftmax(function_node.FunctionNode):

    """Log-softmax activation function."""

    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            -x_type.ndim <= self.axis < x_type.ndim,
        )

    def forward_chainerx(self, xs):
        return chainerx.log_softmax(xs[0], axis=self.axis),

    def forward(self, xs):
        y = _log_softmax(xs[0], axis=self.axis)
        self._x_xp = backend.get_array_module(*xs)
        self._x_shape = xs[0].shape
        self._x_dtype = xs[0].dtype
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, gy):
        y = self.get_retained_outputs()[0]
        return LogSoftmaxGrad(
            self._x_xp, self._x_shape, self._x_dtype, self.axis).apply(
                (y, gy[0]))


class LogSoftmaxGrad(function_node.FunctionNode):

    def __init__(self, x_xp, x_shape, x_dtype, axis):
        self._x_xp = x_xp
        self._x_shape = x_shape
        self._x_dtype = x_dtype
        self.axis = axis

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        xp = self._x_xp
        if xp is cuda.cupy and chainer.should_use_cudnn('>=auto'):
            gx = cudnn.softmax_backward(y, gy, self.axis, _algorithm)
        else:
            gx = gy - xp.exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx,

    def backward(self, indexes, ggx):
        y, gy = self.get_retained_inputs()
        ret = []
        exp_y = chainer.functions.exp(y)
        if 0 in indexes:
            gy_sum = chainer.functions.sum(gy, self.axis, True)
            gy_sum = chainer.functions.broadcast_to(gy_sum, gy.shape)
            g0 = -ggx[0] * exp_y * gy_sum
            ret.append(g0)
        if 1 in indexes:
            # TODO(Kenta Oono): implement it with double-backpropable F.matmul
            a = chainer.functions.sum(ggx[0] * exp_y, self.axis, True)
            a = chainer.functions.broadcast_to(a, gy.shape)
            g1 = ggx[0] - a
            ret.append(g1)
        return ret


def log_softmax(x, axis=1):
    """Channel-wise log-softmax function.

    This function computes its logarithm of softmax along the second axis.
    Let :math:`c = (c_1, c_2, \\dots, c_D)` be the slice of ``x`` along with
    the second axis. For each slice :math:`c`, it computes the logarithm of
    the function :math:`f(c)` defined as

    .. math::
        f(c) = {\\exp(c) \\over \\sum_{d} \\exp(c_d)}.

    This method is theoretically equivalent to ``log(softmax(x))`` but is more
    stable.

    .. note::
        ``log(softmax(x))`` may cause underflow when ``x`` is too small,
        because ``softmax(x)`` may returns ``0``.
        ``log_softmax`` method is more stable.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable.
            A :math:`n`-dimensional (:math:`n \\geq 2`) float array.
        axis (int): The axis along which the softmax is to be computed.

    Returns:
        ~chainer.Variable: Output variable.
        A :math:`n`-dimensional (:math:`n \\geq 2`) float array, which is the
        same shape with x.

    .. seealso:: :func:`~chainer.functions.softmax`

    .. admonition:: Example

        >>> x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        >>> x
        array([[0., 1., 2.],
               [0., 2., 4.]], dtype=float32)
        >>> F.log_softmax(x).array
        array([[-2.407606  , -1.4076059 , -0.4076059 ],
               [-4.1429315 , -2.1429315 , -0.14293146]], dtype=float32)
        >>> np.allclose(F.log_softmax(x).data, F.log(F.softmax(x)).data)
        True

    """
    return LogSoftmax(axis=axis).apply((x,))[0]
