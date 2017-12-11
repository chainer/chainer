from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class LinearInterpolate(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        p_type, x_type, y_type = in_types

        type_check.expect(
            p_type.dtype.kind == 'f',
            x_type.dtype == p_type.dtype,
            y_type.dtype == p_type.dtype,
            p_type.shape == x_type.shape,
            p_type.shape == y_type.shape,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        p, x, y = inputs
        one = p.dtype.type(1)
        return utils.force_array(p * x + (one - p) * y),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        p, x, y = inputs
        return cuda.elementwise(
            'T p, T x, T y', 'T z',
            'z = p * x + (1 - p) * y',
            'linear_interpolate_fwd',
        )(p, x, y),

    def backward(self, indexes, grad_outputs):
        p, x, y = self.get_retained_inputs()
        g, = grad_outputs
        return LinearInterpolateGrad().apply((p, x, y, g))


class LinearInterpolateGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2, 3))
        p, x, y, g = inputs
        pg = p * g
        return (utils.force_array((x - y) * g),
                utils.force_array(pg),
                utils.force_array(g - pg))

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2, 3))
        p, x, y, g = inputs
        return cuda.elementwise(
            'T p, T x, T y, T g', 'T gp, T gx, T gy',
            '''
            gp = (x - y) * g;
            gx = g * p;
            gy = g * (1 - p);
            ''',
            'linear_interpolate_bwd'
        )(p, x, y, g)

    def backward(self, indexes, grad_outputs):
        p, x, y, g = self.get_retained_inputs()
        g0, g1, g2 = grad_outputs
        gp = g * (g1 - g2)
        gx = g * g0
        gy = - gx
        gg = (x - y) * g0 + p * g1 + (1 - p) * g2
        return gp, gx, gy, gg


def linear_interpolate(p, x, y):
    """Elementwise linear-interpolation function.

    This function is defined as

    .. math::

        f(p, x, y) = p x + (1 - p) y.

    Args:
        p (~chainer.Variable): Input variable.
        x (~chainer.Variable): Input variable.
        y (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """

    return LinearInterpolate().apply((p, x, y))[0]
