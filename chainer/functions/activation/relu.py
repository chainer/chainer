import numpy

import chainer
from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cudnn.cudnn.CUDNN_ACTIVATION_RELU


class ReLU(function_node.FunctionNode):

    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.retain_outputs((0,))
        self._use_cudnn = False
        return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('==always') and x[0].flags.c_contiguous:
            self.retain_inputs((0,))
            self._use_cudnn = True
            y = cudnn.activation_forward(x[0], _mode)
        else:
            self._use_cudnn = False
            y = cuda.cupy.maximum(x[0], 0)
        self.retrain_outputs((0,))
        return y,

    def backward(self, indexes, gy):
        y = self.get_retained_outputs()[0]
        if chainer.should_use_cudnn('==always') and self._use_cudnn:
            # The only case to use ReLUGrad3 is compute is done in GPU
            # and _use_cudnn is True.
            x = self.get_retained_inputs()[0]
            return ReLUGrad3().apply((x, y, gy[0]))
        else:
            return ReLUGrad2().apply((y, gy[0]))


class ReLUGrad2(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        b, c = inputs
        y = (b > 0) * c
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return utils.force_array(y, dtype=y.dtype),

    def backward_cpu(self, indexes, gy):
        ret = []
        if 0 in indexes:
            y = self.get_retained_outputs()[0]
            gb = gy * y
            ret.append(gb)
        if 1 in indexes:
            b = self.get_retained_inputs()[0]
            gc = gy * (b > 0)
            ret.append(gc)
        return ret

    def forward_gpu(self, inputs):
        b, c = inputs
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = y > 0 ? gy : (T)0',
            'relu_bwd')(b, c)
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return gx,

    def backward_gpu(self, indexes, gy):
        ret = []
        if 0 in indexes:
            y = self.get_retained_outputs()[0]
            gb = gy * y
            ret.append(gb)
        if 1 in indexes:
            b = self.get_retained_inputs()[0]
            gc = gy * (b > 0)
            ret.append(gc)
        return ret


class ReLUGrad3(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        b, c = inputs
        y = (b > 0) * c
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return y,

    def backward_cpu(self, indexes, gy):
        ret = []
        if 0 in indexes:
            ret.append(None)
        if 1 in indexes:
            y = self.get_retained_outputs()[0]
            gb = gy * y
            ret.append(gb)
        if 2 in indexes:
            b = self.get_retained_inputs()[0]
            gc = gy * (b > 0)
            ret.append(gc)
        return ret

    def forward_gpu(self, inputs):
        a, b, c = inputs
        assert chainer.should_use_cudnn('==always')
        y = cudnn.activation_backward(a, b, c, _mode)
        self.retain_inputs((1,))
        self.retain_outputs((0,))
        return gx,

    def backward_gpu(self, indexes, gy):
        ret = []
        if 0 in indexes:
            ret.append(None)
        if 1 in indexes:
            y = self.get_retained_outputs()[0]
            gb = gy * y
            ret.append(gb)
        if 2 in indexes:
            b = self.get_retained_inputs()[0]
            gc = gy * (b > 0)
            ret.append(gc)
        return ret


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
