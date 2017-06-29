import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_SIGMOID


class Sigmoid(function.Function):

    """Logistic sigmoid function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        half = x[0].dtype.type(0.5)
        y = utils.force_array(numpy.tanh(x[0] * half) * half + half)
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, inputs):
        x = inputs[0]
        if (chainer.should_use_cudnn('==always') and
                x.flags.c_contiguous and
                (_cudnn_version >= 3000 or x.dtype != numpy.float16)):
            y = cudnn.activation_forward(x, _mode)
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x * 0.5) * 0.5 + 0.5',
                'sigmoid_fwd')(x)
            self.retain_inputs(())
        self.retain_outputs((0,))
        return y,

    def backward_cpu(self, x, gy):
        one = x[0].dtype.type(1)
        y = self.output_data[0]
        return utils.force_array(gy[0] * y * (one - y)),

    def backward_gpu(self, inputs, grads):
        x = inputs[0]
        gy = grads[0]
        y = self.output_data[0]
        if (chainer.should_use_cudnn('==always') and
                gy.flags.c_contiguous and
                x is not None and
                x.flags.c_contiguous and
                (_cudnn_version >= 3000 or x.dtype != numpy.float16)):
            gx = cudnn.activation_backward(x, y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(y, gy)
        return gx,


def sigmoid(x):
    """Element-wise sigmoid logistic function.

     .. math:: f(x)=(1 + \\exp(-x))^{-1}.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        It maps the input values into the range of :math:`[0, 1]`.

        >>> x = np.arange(-2, 3, 2).astype('f')
        >>> x
        array([-2.,  0.,  2.], dtype=float32)
        >>> F.sigmoid(x).data
        array([ 0.11920291,  0.5       ,  0.88079709], dtype=float32)

    """
    return Sigmoid()(x)
