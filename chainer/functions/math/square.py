from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Square(function_node.FunctionNode):

    @property
    def label(self):
        return 'square'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.square(x[0], dtype=x[0].dtype)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        gx = gy[0] * 2.0 * x
        return gx,


def square(x):
    """Elementwise square function.

    .. math::
       y_i = x_i ^ 2.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

    Returns:
        ~chainer.Variable: Output variable.
        A :math:`(s_1, s_2, ..., s_N)` -shaped float array.

    .. admonition:: Example

        >>> x = np.arange(6).reshape(2,3).astype(np.float32)
        >>> x
        array([[0., 1., 2.],
               [3., 4., 5.]], dtype=float32)
        >>> y = F.square(x)
        >>> y.shape
        (2, 3)
        >>> y.array
        array([[ 0.,  1.,  4.],
               [ 9., 16., 25.]], dtype=float32)

    """
    return Square().apply((x,))[0]
