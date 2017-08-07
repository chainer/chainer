import numpy

import chainer
from chainer import cuda
from chainer import function_node
from chainer.functions.array import cast
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_SIGMOID


class Sigmoid(function_node.FunctionNode):

    """Logistic sigmoid function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x = inputs[0]
        half = x.dtype.type(0.5)
        y = utils.force_array(numpy.tanh(x * half) * half + half)
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, inputs):
        x = inputs[0]
        if chainer.should_use_cudnn('==always') and x.flags.c_contiguous:
            y = cudnn.activation_forward(x, _mode)
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x * 0.5) * 0.5 + 0.5',
                'sigmoid_fwd')(x)
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        print(self.get_retained_inputs()[0])
        if isinstance(self.get_retained_inputs()[0].data, cuda.ndarray):
            return self.backward_gpu(indexes, grad_outputs)
        else:
            return self.backward_cpu(indexes, grad_outputs)

    def backward_cpu(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        y, = self.get_retained_outputs()
        gy, = grad_outputs
        one = x.dtype.type(1)
        ret = []
        if 0 in indexes:
            gx = utils.force_array(gy * y * (one - y))
            ret.append(cast.cast(gx, x.dtype))

        return ret

    def backward_gpu(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0].data
        y = self.get_retained_outputs()[0].data
        gy = grad_outputs[0].data
        if (chainer.should_use_cudnn('==always') and gy.flags.c_contiguous
                and x is not None and x.flags.c_contiguous):
            gx = cudnn.activation_backward(x, y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(y, gy)
        ret = []
        if 0 in indexes:
            ret.append(cast.cast(gx, x.dtype))
        return ret


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
        variable([ 0.11920291,  0.5       ,  0.88079709])

    """
    y, = Sigmoid().apply((x,))
    return y
