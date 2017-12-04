import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.cuda.cudnn.CUDNN_ACTIVATION_RELU


class ReLU(function_node.FunctionNode):

    """Rectified Linear Unit."""

    _use_cudnn = False

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.retain_outputs((0,))
        return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('==always') and x[0].flags.c_contiguous:
            # cupy.activation_backward requires the input.
            # So, we retain it for backward computation.
            self.retain_inputs((0,))
            self._use_cudnn = True
            y = cudnn.activation_forward(x[0], _mode)
        else:
            y = cuda.cupy.maximum(x[0], 0)
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, gy):
        y = self.get_retained_outputs()[0]
        if chainer.should_use_cudnn('==always') and self._use_cudnn:
            x = self.get_retained_inputs()[0]
            return ReLUGrad3(x, y).apply((gy[0],))
        else:
            return ReLUGrad2(y).apply((gy[0],))


def _heaviside(x):
    return (x > 0).astype(x.dtype)


class ReLUGrad2(function_node.FunctionNode):
    """Computes the gradient of the ReLU function.

    This function takes 2 variables b and c, and
    computes f(b, c) = sign(b) * c with backpropagation
    where operations are done in elementwise manner
    and sign(x) = 1 when x > 0 is positive and 0 otherwise.

    As the gradient of f with respect to b is 0,
    we do not backpropagate errors toward b for computational efficiency.
    """

    def __init__(self, b):
        super(ReLUGrad2, self).__init__()
        self.b = b.data

    def forward_cpu(self, inputs):
        y = (self.b > 0) * inputs[0]
        return utils.force_array(y, dtype=y.dtype),

    def forward_gpu(self, inputs):
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = y > 0 ? gy : (T)0',
            'relu_bwd')(self.b, inputs[0])
        return gx,

    def backward(self, indexes, gy):
        return gy[0] * _heaviside(self.b),


class ReLUGrad3(function_node.FunctionNode):
    """Computes the gradient of the ReLU function.

    This function takes 3 variables a, b, and c, and
    computes f(a, b, c) = sign(b) * c with backpropagation
    where operations are dones in elementwise manner
    and sign(x) = 1 if x > 0 is positive and 0 otherwise.

    As the gradient of f with respect to a and b are 0,
    we do not backpropagate errors toward them for computational efficiency.
    """

    def __init__(self, a, b):
        super(ReLUGrad3, self).__init__()
        self.a = a.data
        self.b = b.data

    def forward_cpu(self, inputs):
        return (self.b > 0) * inputs[0],

    def forward_gpu(self, inputs):
        assert chainer.should_use_cudnn('==always')
        return cudnn.activation_backward(self.a, self.b, inputs[0], _mode),

    def backward(self, indexes, gy):
        return gy[0] * _heaviside(self.b),


def relu(x):
    """Rectified Linear Unit function.

    .. math:: f(x)=\\max(0, x).

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> np.any(x < 0)
        True
        >>> y = F.relu(x)
        >>> np.any(y.data < 0)
        False
        >>> y.shape
        (3, 2)

    """
    y, = ReLU().apply((x,))
    return y
