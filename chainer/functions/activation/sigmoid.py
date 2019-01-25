import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.libcudnn.CUDNN_ACTIVATION_SIGMOID


class Sigmoid(function_node.FunctionNode):

    """Logistic sigmoid function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x = inputs[0]
        half = x.dtype.type(0.5)
        y = utils.force_array(numpy.tanh(x * half) * half + half)
        self.retain_outputs((0,))
        self._use_cudnn = False
        return y,

    def forward_gpu(self, inputs):
        x = inputs[0]
        if chainer.should_use_cudnn('==always') and x.flags.c_contiguous:
            y = cudnn.activation_forward(x, _mode)
            self.retain_inputs((0,))
            self._use_cudnn = True
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x * 0.5) * 0.5 + 0.5',
                'sigmoid_fwd')(x)
            self._use_cudnn = False

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        if self._use_cudnn:
            x = self.get_retained_inputs()[0].data
        else:
            x = None
        y = self.get_retained_outputs()[0]
        gy, = grad_outputs
        return SigmoidGrad((x,)).apply((y, gy))


class SigmoidGrad(function_node.FunctionNode):

    """Logistic sigmoid gradient function."""

    def __init__(self, inputs):
        super(SigmoidGrad, self).__init__()
        self.x = inputs[0]

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('y', 'gy'))
        type_check.expect(in_types[0].dtype.kind == 'f')
        type_check.expect(in_types[1].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        one = y.dtype.type(1)
        return utils.force_array(gy * y * (one - y)),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        if (chainer.should_use_cudnn('==always') and gy.flags.c_contiguous and
                self.x is not None and self.x.flags.c_contiguous):
            gx = cudnn.activation_backward(self.x, y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(y, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        ggx, = grad_outputs
        return ggx * gy * (1 - 2 * y), ggx * y * (1 - y)


def sigmoid(x):
    """Element-wise sigmoid logistic function.

     .. math:: f(x)=(1 + \\exp(-x))^{-1}.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        It maps the input values into the range of :math:`[0, 1]`.

        >>> x = np.arange(-2, 3, 2).astype(np.float32)
        >>> x
        array([-2.,  0.,  2.], dtype=float32)
        >>> F.sigmoid(x).array
        array([0.11920291, 0.5       , 0.8807971 ], dtype=float32)

    """
    y, = Sigmoid().apply((x,))
    return y
