import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx


if cuda.available:
    _relu_grad2_kernel = cuda.elementwise(
        'T y, T gy', 'T gx',
        'gx = y > 0 ? gy : (T)0', 'relu_bwd')
if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.cuda.cudnn.CUDNN_ACTIVATION_RELU  # type: ignore


class ReLU(function_node.FunctionNode):

    """Rectified Linear Unit."""

    _use_cudnn = False

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_chainerx(self, inputs):
        x, = inputs
        return chainerx.maximum(x, 0),

    def forward_cpu(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            return self.forward_ideep(inputs)

        x, = inputs
        y = numpy.maximum(x, 0, dtype=x.dtype)
        self.retain_outputs((0,))
        return utils.force_array(y),

    def forward_ideep(self, inputs):
        x, = inputs
        y = intel64.ideep.relu.Forward(intel64.ideep.array(x))
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, inputs):
        x, = inputs
        if chainer.should_use_cudnn('>=auto') and x.flags.c_contiguous:
            self._use_cudnn = True
            y = cudnn.activation_forward(x, _mode)
        else:
            y = cuda.cupy.maximum(x, 0, dtype=x.dtype)
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        y, = self.get_retained_outputs()

        if self._use_cudnn and chainer.should_use_cudnn('>=auto'):
            # cuDNN implementation
            return ReLUGradCudnn(y.array).apply((gy,))

        # Generic implementation
        return ReLUGrad2(y.array).apply((gy,))


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
        self.b = b

    def forward_cpu(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            return self.forward_ideep(inputs)

        gy, = inputs
        gx = gy * (self.b > 0)
        return utils.force_array(gx, dtype=gy.dtype),

    def forward_ideep(self, inputs):
        gy, = inputs
        gx = intel64.ideep.relu.Backward(
            intel64.ideep.array(self.b),
            intel64.ideep.array(gy))
        return gx,

    def forward_gpu(self, inputs):
        gx = _relu_grad2_kernel(self.b, inputs[0])
        return gx,

    def backward(self, indexes, grad_outputs):
        return ReLUGrad2(self.b).apply(grad_outputs)


class ReLUGradCudnn(function_node.FunctionNode):
    """Computes the gradient of the ReLU function.

    This function takes 3 variables a, b, and c, and
    computes f(a, b, c) = sign(b) * c with backpropagation
    where operations are dones in elementwise manner
    and sign(x) = 1 if x > 0 is positive and 0 otherwise.

    As the gradient of f with respect to a and b are 0,
    we do not backpropagate errors toward them for computational efficiency.
    """

    def __init__(self, y):
        super(ReLUGradCudnn, self).__init__()
        self.y = y

    def forward(self, inputs):
        gy, = inputs
        return cudnn.activation_backward(self.y, self.y, gy, _mode),

    def backward(self, indexes, grad_outputs):
        return ReLUGrad2(self.y).apply(grad_outputs)


def relu(x):
    """Rectified Linear Unit function.

    .. math:: f(x)=\\max(0, x).

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        >>> np.any(x < 0)
        True
        >>> y = F.relu(x)
        >>> np.any(y.array < 0)
        False
        >>> y.shape
        (3, 2)

    """
    y, = ReLU().apply((x,))
    return y
