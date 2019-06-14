from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class LinearInterpolate(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('p', 'x', 'y'))
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
        gz, = grad_outputs
        return LinearInterpolateGrad().apply((p, x, y, gz))


class LinearInterpolateGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2, 3))
        p, x, y, gz = inputs
        pg = p * gz
        return (utils.force_array((x - y) * gz),
                utils.force_array(pg),
                utils.force_array(gz - pg))

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2, 3))
        p, x, y, gz = inputs
        return cuda.elementwise(
            'T p, T x, T y, T gz', 'T gp, T gx, T gy',
            '''
            gp = (x - y) * gz;
            gx = gz * p;
            gy = gz * (1 - p);
            ''',
            'linear_interpolate_bwd'
        )(p, x, y, gz)

    def backward(self, indexes, grad_outputs):
        p, x, y, gz = self.get_retained_inputs()
        ggp, ggx, ggy = grad_outputs
        gp = gz * (ggx - ggy)
        gx = gz * ggp
        gy = - gx
        ggz = (x - y) * ggp + p * ggx + (1 - p) * ggy
        return gp, gx, gy, ggz


def linear_interpolate(p, x, y):
    """Elementwise linear-interpolation function.

    This function is defined as

    .. math::

        f(p, x, y) = p x + (1 - p) y.

    Args:
        p (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        y (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """

    return LinearInterpolate().apply((p, x, y))[0]
