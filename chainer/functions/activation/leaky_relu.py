from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer.utils import type_check


_kern = None


def _get_kern():
    global _kern
    if _kern is None:
        _kern = cuda.elementwise(
            'T cond, T x, T slope', 'T y',
            'y = cond <= 0 ? (T)(slope * x) : x', 'lrelu')
    return _kern


class LeakyReLU(function_node.FunctionNode):

    """Leaky rectifier unit."""

    def __init__(self, slope=0.2):
        self.slope = slope

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            return self.forward_ideep(inputs)

        x, = inputs
        y = x.copy()
        y[x <= 0] *= self.slope
        if self.slope >= 0:
            self.retain_outputs((0,))
        else:
            self.retain_inputs((0,))
        return y,

    def forward_ideep(self, inputs):
        x, = inputs
        y = intel64.ideep.relu.Forward(
            intel64.ideep.array(x), self.slope)
        if self.slope >= 0:
            self.retain_outputs((0,))
        else:
            self.retain_inputs((0,))
        return y,

    def forward_gpu(self, inputs):
        x, = inputs
        y = _get_kern()(x, x, self.slope)
        if self.slope >= 0:
            self.retain_outputs((0,))
        else:
            self.retain_inputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        if self.slope >= 0:
            cond = self.get_retained_outputs()[0].array
        else:
            cond = self.get_retained_inputs()[0].array
        return _LeakyReLUGrad(cond, self.slope).apply(grad_outputs)


class _LeakyReLUGrad(function_node.FunctionNode):

    def __init__(self, cond, slope):
        self.cond = cond
        self.slope = slope

    def forward_cpu(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            return self.forward_ideep(inputs)

        gy, = inputs
        gy = gy.copy()
        gy[self.cond <= 0] *= self.slope
        return gy,

    def forward_ideep(self, inputs):
        gy, = inputs
        gy = intel64.ideep.relu.Backward(
            intel64.ideep.array(self.cond),
            intel64.ideep.array(gy), self.slope)
        return gy,

    def forward_gpu(self, inputs):
        gy, = inputs
        gy = _get_kern()(self.cond, gy, self.slope)
        return gy,

    def backward(self, indexes, grad_outputs):
        return _LeakyReLUGrad(self.cond, self.slope).apply(grad_outputs)


def leaky_relu(x, slope=0.2):
    """Leaky Rectified Linear Unit function.

    This function is expressed as

    .. math::

        f(x) = \\left \\{ \\begin{array}{ll}
        x  & {\\rm if}~ x \\ge 0 \\\\
        ax & {\\rm if}~ x < 0,
        \\end{array} \\right.

    where :math:`a` is a configurable slope value.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        slope (float): Slope value :math:`a`.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        >>> x
        array([[-1.,  0.],
               [ 2., -3.],
               [-2.,  1.]], dtype=float32)
        >>> F.leaky_relu(x, slope=0.2).array
        array([[-0.2,  0. ],
               [ 2. , -0.6],
               [-0.4,  1. ]], dtype=float32)

    """
    return LeakyReLU(slope).apply((x,))[0]
