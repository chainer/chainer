import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.cuda.cudnn.CUDNN_ACTIVATION_RELU


class ReLU(function_node.FunctionNode):

    """Rectified Linear Unit."""

    _use_cudnn = False
    _use_ideep = False

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
            self._use_ideep = True
            return self.forward_ideep(inputs)

        x, = inputs
        self.retain_outputs((0,))
        return utils.force_array(numpy.maximum(x, 0, dtype=x.dtype)),

    def forward_ideep(self, inputs):
        x, = inputs
        self.retain_inputs((0,))
        self.retain_outputs((0,))

        y = intel64.ideep.relu.Forward(intel64.ideep.array(x))
        return y,

    def forward_gpu(self, inputs):
        x, = inputs
        if chainer.should_use_cudnn('==always') and x.flags.c_contiguous:
            # cupy.activation_backward requires the input.
            # So, we retain it for backward computation.
            self.retain_inputs((0,))
            self._use_cudnn = True
            y = cudnn.activation_forward(x, _mode)
        else:
            y = cuda.cupy.maximum(x, 0)
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        y, = self.get_retained_outputs()
        if self._use_ideep:
            # iDeep implementation
            x, = self.get_retained_inputs()
            return ReLUGradIdeep(x, y).apply((gy,))
        if chainer.should_use_cudnn('==always') and self._use_cudnn:
            # cuDNN implementation
            x, = self.get_retained_inputs()
            return ReLUGradCudnn(x, y).apply((gy,))
        # Generic implementation
        return ReLUGrad2(y).apply((gy,))


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


class ReLUGrad3Base(function_node.FunctionNode):
    """Computes the gradient of the ReLU function.

    This function takes 3 variables a, b, and c, and
    computes f(a, b, c) = sign(b) * c with backpropagation
    where operations are dones in elementwise manner
    and sign(x) = 1 if x > 0 is positive and 0 otherwise.

    As the gradient of f with respect to a and b are 0,
    we do not backpropagate errors toward them for computational efficiency.
    """

    def __init__(self, x, y):
        super(ReLUGrad3Base, self).__init__()
        self.x = x.data
        self.y = y.data

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ggx = gy * _heaviside(self.y)
        return ggx,


class ReLUGradCudnn(ReLUGrad3Base):

    def forward(self, inputs):
        assert chainer.should_use_cudnn('==always')
        gy, = inputs
        return cudnn.activation_backward(self.x, self.y, gy, _mode),


class ReLUGradIdeep(ReLUGrad3Base):

    def forward(self, inputs):
        gy, = inputs
        ggx = intel64.ideep.relu.Backward(
            intel64.ideep.array(self.x),
            intel64.ideep.array(gy))
        return ggx,


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

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
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
