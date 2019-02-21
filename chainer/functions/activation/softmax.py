import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _algorithm = cuda.libcudnn.CUDNN_SOFTMAX_ACCURATE


class Softmax(function_node.FunctionNode):

    """Softmax activation function."""

    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            -x_type.ndim <= self.axis < x_type.ndim,
        )

    def forward(self, x):
        xp = backend.get_array_module(*x)
        if xp is cuda.cupy and chainer.should_use_cudnn('>=auto'):
            y = cudnn.softmax_forward(x[0], self.axis, _algorithm)
        else:
            y = x[0] - x[0].max(axis=self.axis, keepdims=True)
            xp.exp(y, out=y)
            y /= y.sum(axis=self.axis, keepdims=True)

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        y = self.get_retained_outputs()[0]
        gy, = grad_outputs
        return _SoftmaxGrad(self.axis).apply((y, gy))


class _SoftmaxGrad(function_node.FunctionNode):

    def __init__(self, axis):
        self.axis = axis

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        xp = backend.get_array_module(*y)
        if xp is cuda.cupy and chainer.should_use_cudnn('>=auto'):
            gx = cudnn.softmax_backward(y, gy, self.axis, _algorithm)
        else:
            gx = y * gy
            sumdx = gx.sum(axis=self.axis, keepdims=True)
            gx -= y * sumdx

        return gx,

    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        ggx, = grad_outputs
        gs = chainer.functions.sum(ggx * y, axis=self.axis, keepdims=True)
        ga = ggx - chainer.functions.broadcast_to(gs, gy.shape)
        ret = []
        if 0 in indexes:
            s = chainer.functions.broadcast_to(chainer.functions.sum(
                y * gy, axis=self.axis, keepdims=True), gy.shape)
            gy2 = ga * gy - ggx * s
            ret.append(gy2)
        if 1 in indexes:
            ggy = ga * y
            ret.append(ggy)
        return tuple(ret)


def softmax(x, axis=1):
    """Softmax function.

    This function computes its softmax along an axis. Let
    :math:`c = (c_1, c_2, \\dots, c_D)` be the slice of ``x`` along with
    the axis. For each slice :math:`c`, it computes the function :math:`f(c)`
    defined as :math:`f(c)={\\exp(c) \\over \\sum_{d} \\exp(c_d)}`.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable.
            A :math:`n`-dimensional (:math:`n \\geq 2`) float array.
        axis (int): The axis along which the softmax is to be computed.

    Returns:
        ~chainer.Variable: Output variable.
        A :math:`n`-dimensional (:math:`n \\geq 2`) float array, which is the
        same shape with x.

    .. admonition:: Example

        >>> x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        >>> x
        array([[0., 1., 2.],
               [0., 2., 4.]], dtype=float32)
        >>> y = F.softmax(x, axis=1)
        >>> y.array
        array([[0.09003057, 0.24472848, 0.66524094],
               [0.01587624, 0.11731043, 0.86681336]], dtype=float32)
        >>> F.sum(y, axis=1).array
        array([1., 1.], dtype=float32)

    """
    return Softmax(axis=axis).apply((x,))[0]
