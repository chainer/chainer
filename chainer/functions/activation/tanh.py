import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.libcudnn.CUDNN_ACTIVATION_TANH


class Tanh(function_node.FunctionNode):

    """Hyperbolic tangent function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_chainerx(self, x):
        return chainerx.tanh(x[0]),

    def forward_cpu(self, x):
        y = utils.force_array(numpy.tanh(x[0]))
        self.retain_outputs((0,))
        self._use_cudnn = False
        return y,

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('==always') and x[0].flags.c_contiguous:
            y = cudnn.activation_forward(x[0], _mode)
            self.retain_inputs((0,))
            self._use_cudnn = True
        else:
            y = cuda.cupy.empty_like(x[0])
            cuda.cupy.tanh(x[0], out=y)
            self._use_cudnn = False

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        if self._use_cudnn:
            x = self.get_retained_inputs()[0].data
        else:
            x = None
        y = self.get_retained_outputs()[0]
        gy = grad_outputs[0]
        return TanhGrad(x).apply((y, gy))


class TanhGrad(function_node.FunctionNode):

    def __init__(self, x):
        super(TanhGrad, self).__init__()
        # The original input `x` is only required for cuDNN.
        # If it is None, this class does not use cuDNN.
        # Note that x must be c-contiguous and it is checked
        # in Tanh.forward_gpu.
        self.x = x

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        one = y.dtype.type(1)
        return utils.force_array(gy * (one - y * y)),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        if (chainer.should_use_cudnn('==always') and
                self.x is not None and gy.flags.c_contiguous):
            gx = cudnn.activation_backward(self.x, y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * (1 - y * y)',
                'tanh_bwd')(y, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        ggx = grad_outputs[0]

        y_mul_ggx = y * ggx
        grad_y = -2 * gy * y_mul_ggx
        ggy = ggx - y * y_mul_ggx
        return grad_y, ggy


def tanh(x):
    """Elementwise hyperbolic tangent function.

     .. math:: f(x)=\\tanh(x).

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.arange(-1, 4, 2).astype(np.float32)
        >>> x
        array([-1.,  1.,  3.], dtype=float32)
        >>> F.tanh(x).array
        array([-0.7615942,  0.7615942,  0.9950548], dtype=float32)

    """
    return Tanh().apply((x,))[0]
