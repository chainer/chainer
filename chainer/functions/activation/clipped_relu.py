import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.cuda.cudnn.CUDNN_ACTIVATION_CLIPPED_RELU


class ClippedReLU(function_node.FunctionNode):

    """Clipped Rectifier Unit function.

    Clipped ReLU is written as
    :math:`ClippedReLU(x, z) = \\min(\\max(0, x), z)`,
    where :math:`z(>0)` is a parameter to cap return value of ReLU.

    """

    _use_cudnn = False

    def __init__(self, z):
        if not isinstance(z, float):
            raise TypeError('z must be float value')
        # z must be positive.
        assert z > 0
        self.cap = z

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        return utils.force_array(numpy.minimum(numpy.maximum(0, x), self.cap),
                                 x.dtype),

    def forward_gpu(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        if chainer.should_use_cudnn('==always') and x.flags.c_contiguous:
            self._use_cudnn = True
            y = cudnn.activation_forward(x, _mode, self.cap)
            self.retain_outputs((0,))
        else:
            return cuda.elementwise(
                'T x, T cap', 'T y', 'y = min(max(x, (T)0), cap)',
                'clipped_relu_fwd')(x, self.cap),
        return y,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        if chainer.should_use_cudnn('==always') and self._use_cudnn:
            y = self.get_retained_outputs()[0]
            return ClippedReLUGrad3(x.data, y.data, self.cap).apply(
                grad_outputs)
        else:
            return ClippedReLUGrad2(x.data, self.cap).apply(grad_outputs)


class ClippedReLUGrad2(function_node.FunctionNode):

    """Clipped Rectifier Unit gradient function."""

    def __init__(self, x, z):
        self.x = x
        self.cap = z

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        gy, = inputs
        return utils.force_array(
            gy * (0 < self.x) * (self.x < self.cap), self.x.dtype),

    def forward_gpu(self, inputs):
        gy, = inputs
        gx = cuda.elementwise(
            'T x, T gy, T z', 'T gx',
            'gx = ((x > 0) & (x < z)) ? gy : (T)0',
            'clipped_relu_bwd')(self.x, gy, self.cap)
        return gx,

    def backward(self, indexes, grad_outputs):
        return ClippedReLUGrad2(self.x, self.cap).apply(grad_outputs)


class ClippedReLUGrad3(function_node.FunctionNode):

    """Clipped Rectifier Unit gradient function."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.cap = z

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        gy, = inputs
        return utils.force_array(
            gy * (0 < self.x) * (self.x < self.cap), self.x.dtype),

    def forward_gpu(self, inputs):
        assert chainer.should_use_cudnn('==always')
        return cudnn.activation_backward(self.x, self.y, inputs[0], _mode,
                                         self.cap),

    def backward(self, indexes, grad_outputs):
        return ClippedReLUGrad3(self.x, self.y, self.cap).apply(grad_outputs)


def clipped_relu(x, z=20.0):
    """Clipped Rectifier Unit function.

    For a clipping value :math:`z(>0)`, it computes

    .. math:: \\text{ClippedReLU}(x, z) = \\min(\\max(0, x), z).

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.
        z (float): Clipping value. (default = 20.0)

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_n)`-shaped float array.

    .. admonition:: Example

        >>> x = np.random.uniform(-100, 100, (10, 20)).astype(np.float32)
        >>> z = 10.0
        >>> np.any(x < 0)
        True
        >>> np.any(x > z)
        True
        >>> y = F.clipped_relu(x, z=z)
        >>> np.any(y.data < 0)
        False
        >>> np.any(y.data > z)
        False

    """
    y, = ClippedReLU(z).apply((x,))
    return y
