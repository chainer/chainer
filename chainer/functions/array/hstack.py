import numpy
import six

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Hstack(function_node.FunctionNode):

    """Concatenate multiple tensors horizontally (column wise)."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check._argname((in_types[0],), ('x0',))

        ndim = type_check.eval(in_types[0].ndim)
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check._argname((in_types[i],), ('x{}'.format(i),))
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                continue
            for d in six.moves.range(0, ndim):
                if d == 1:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        xp = backend.get_array_module(*xs)
        return xp.hstack(xs),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ndim = len(self.inputs[0].shape)
        if len(self.inputs) == 1:
            if ndim == 0:
                return gy.reshape(()),
            return gy,

        if ndim == 0:
            gx = chainer.functions.split_axis(gy, len(self.inputs), 0)
            return [g.reshape(()) for g in gx]

        axis = 0 if ndim == 1 else 1
        sizes = numpy.array([x.shape[axis] for x in self.inputs[:-1]]).cumsum()
        return chainer.functions.split_axis(gy, sizes, axis)


def hstack(xs):
    """Concatenate variables horizontally (column wise).

    Args:
        xs (list of :class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be concatenated. The variables must have the
            same ``ndim``. When the variables have the second axis (i.e.
            :math:`ndim \\geq 2`), the variables must have the same shape
            along all but the second axis. When the variables do not have the
            second axis(i.e. :math:`ndim < 2`), the variables need not to have
            the same shape.

    Returns:
        ~chainer.Variable:
            Output variable. When the input variables have the second axis
            (i.e. :math:`ndim \\geq 2`), the shapes of inputs and output are
            the same along all but the second axis. The length of second axis
            is the sum of the lengths of inputs' second axis.
            When the variables do not have the second axis (i.e.
            :math:`ndim < 2`), the shape of output is ``(N, )`` (``N`` is the
            sum of the input variables' size).

    .. admonition:: Example

        >>> x1 = np.array((1, 2, 3))
        >>> x1.shape
        (3,)
        >>> x2 = np.array((2, 3, 4))
        >>> x2.shape
        (3,)
        >>> y = F.hstack((x1, x2))
        >>> y.shape
        (6,)
        >>> y.array
        array([1, 2, 3, 2, 3, 4])
        >>> x1 = np.arange(0, 12).reshape(3, 4)
        >>> x1.shape
        (3, 4)
        >>> x1
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> x2 = np.arange(12, 18).reshape(3, 2)
        >>> x2.shape
        (3, 2)
        >>> x2
        array([[12, 13],
               [14, 15],
               [16, 17]])
        >>> y = F.hstack([x1, x2])
        >>> y.shape
        (3, 6)
        >>> y.array
        array([[ 0,  1,  2,  3, 12, 13],
               [ 4,  5,  6,  7, 14, 15],
               [ 8,  9, 10, 11, 16, 17]])

    """

    y, = Hstack().apply(xs)
    return y
