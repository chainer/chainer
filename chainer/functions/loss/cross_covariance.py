import chainer
from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class CrossCovariance(function_node.FunctionNode):

    """Cross-covariance loss."""

    def __init__(self, reduce='half_squared_sum'):
        self.y_centered = None
        self.z_centered = None
        self.covariance = None

        if reduce not in ('half_squared_sum', 'no'):
            raise ValueError(
                'Only \'half_squared_sum\' and \'no\' are valid '
                'for \'reduce\', but \'%s\' is given' % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('y', 'z'))
        y_type, z_type = in_types

        type_check.expect(
            y_type.dtype.kind == 'f',
            y_type.dtype == z_type.dtype,
            y_type.ndim == 2,
            z_type.ndim == 2,
            y_type.shape[0] == z_type.shape[0]
        )

    def forward(self, inputs):
        y, z = inputs
        self.retain_inputs((0, 1))

        y_centered = y - y.mean(axis=0, keepdims=True)
        z_centered = z - z.mean(axis=0, keepdims=True)
        covariance = y_centered.T.dot(z_centered)
        covariance /= len(y)

        if self.reduce == 'half_squared_sum':
            xp = backend.get_array_module(*inputs)
            cost = xp.vdot(covariance, covariance)
            cost *= y.dtype.type(0.5)
            return utils.force_array(cost),
        else:
            return covariance,

    def backward(self, indexes, grad_outputs):
        y, z = self.get_retained_inputs()
        gcost, = grad_outputs

        y_mean = chainer.functions.mean(y, axis=0, keepdims=True)
        z_mean = chainer.functions.mean(z, axis=0, keepdims=True)
        y_centered = y - chainer.functions.broadcast_to(y_mean, y.shape)
        z_centered = z - chainer.functions.broadcast_to(z_mean, z.shape)
        gcost_div_n = gcost / gcost.dtype.type(len(y))

        ret = []
        if self.reduce == 'half_squared_sum':
            covariance = chainer.functions.matmul(y_centered.T, z_centered)
            covariance /= len(y)
            if 0 in indexes:
                gy = chainer.functions.matmul(z_centered, covariance.T)
                gy *= chainer.functions.broadcast_to(gcost_div_n, gy.shape)
                ret.append(gy)
            if 1 in indexes:
                gz = chainer.functions.matmul(y_centered, covariance)
                gz *= chainer.functions.broadcast_to(gcost_div_n, gz.shape)
                ret.append(gz)
        else:
            if 0 in indexes:
                gy = chainer.functions.matmul(z_centered, gcost_div_n.T)
                ret.append(gy)
            if 1 in indexes:
                gz = chainer.functions.matmul(y_centered, gcost_div_n)
                ret.append(gz)
        return ret


def cross_covariance(y, z, reduce='half_squared_sum'):
    """Computes the sum-squared cross-covariance penalty between ``y`` and ``z``

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the covariant
    matrix that has as many rows (resp. columns) as the dimension of
    ``y`` (resp.z).
    If it is ``'half_squared_sum'``, it holds the half of the
    Frobenius norm (i.e. L2 norm of a matrix flattened to a vector)
    of the covarianct matrix.

    Args:
        y (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a matrix where the first dimension
            corresponds to the batches.
        z (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a matrix where the first dimension
            corresponds to the batches.
        reduce (str): Reduction option. Its value must be either
            ``'half_squared_sum'`` or ``'no'``.
            Otherwise, :class:`ValueError` is raised.

    Returns:
        Variable:
            A variable holding the cross covariance loss.
            If ``reduce`` is ``'no'``, the output variable holds
            2-dimensional array matrix of shape ``(M, N)`` where
            ``M`` (resp. ``N``) is the number of columns of ``y``
            (resp. ``z``).
            If it is ``'half_squared_sum'``, the output variable
            holds a scalar value.

    .. note::

       This cost can be used to disentangle variables.
       See https://arxiv.org/abs/1412.6583v3 for details.

    """
    return CrossCovariance(reduce).apply((y, z))[0]
