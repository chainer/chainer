from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class SquaredError(function_node.FunctionNode):

    """Squared error function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        x0, x1 = inputs
        diff = x0 - x1
        self.retain_inputs((0, 1))
        return utils.force_array(diff * diff, dtype=x0.dtype),

    def backward(self, indexes, grad_outputs):
        x0, x1 = self.get_retained_inputs()
        gy, = grad_outputs
        gx = gy * 2 * (x0 - x1)
        return gx, -gx


def squared_error(x0, x1):
    """Squared error function.

    This function computes the squared error between two variables:

    .. math::

        (x_0 - x_1)^2

    where operation is done in elementwise manner.
    Note that the error is not scaled by 1/2:

    Args:
        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable:
            A variable holding an array representing the squared error of
            two inputs.

    .. note::

        :func:`~chainer.functions.squared_error` and
        :func:`~chainer.functions.squared_difference` are identical functions,
        aside from the different argument names.
        They are both kept for backward compatibility.

    .. seealso:: :func:`~chainer.functions.squared_difference`

    .. admonition:: Example

        >>> x1 = np.arange(6).astype(np.float32)
        >>> x1
        array([0., 1., 2., 3., 4., 5.], dtype=float32)
        >>> x2 = np.array([5, 4, 3, 2, 1, 0]).astype(np.float32)
        >>> x2
        array([5., 4., 3., 2., 1., 0.], dtype=float32)
        >>> y = F.squared_error(x1, x2)
        >>> y.shape
        (6,)
        >>> y.array
        array([25.,  9.,  1.,  1.,  9., 25.], dtype=float32)

    .. seealso:: :func:`~chainer.functions.squared_difference`

    """
    return SquaredError().apply((x0, x1))[0]


def squared_difference(x1, x2):
    """Squared difference function.

    This functions is identical to :func:`~chainer.functions.squared_error`
    except for the names of the arguments.

    .. seealso:: :func:`~chainer.functions.squared_error`

    """
    return squared_error(x1, x2)
