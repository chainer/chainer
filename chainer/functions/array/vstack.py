import numpy
import six

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Vstack(function_node.FunctionNode):

    """Concatenate multiple tensors vertically (row wise)."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = type_check.eval(in_types[0].ndim)
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in six.moves.range(1, ndim):
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        xp = backend.get_array_module(*xs)
        return xp.vstack(xs),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ndim = len(self.inputs[0].shape)
        if len(self.inputs) == 1:
            if ndim <= 1:
                return gy.reshape(self.inputs[0].shape),
            return gy,

        if ndim <= 1:
            gxs = chainer.functions.split_axis(gy, len(self.inputs), 0)
            return [gx.reshape(self.inputs[0].shape) for gx in gxs]

        sizes = numpy.array([x.shape[0] for x in self.inputs[:-1]]).cumsum()
        return chainer.functions.split_axis(gy, sizes, 0)


def vstack(xs):
    """Concatenate variables vertically (row wise).

    Args:
        xs (list of :class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be concatenated. The variables must have the
            same ``ndim``. When the variables have the second axis (i.e.
            :math:`ndim \\geq 2`), the variables must have the same shape
            along all but the first axis. When the variables do not have the
            second axis(i.e. :math:`ndim < 2`), the variables must have the
            same shape.

    Returns:
        ~chainer.Variable:
            Output variable. When the input variables have the second axis
            (i.e. :math:`ndim \\geq 2`), the shapes of inputs and output are
            the same along all but the first axis. The length of first axis
            is the sum of the lengths of inputs' first axis.
            When the variables do not have the second axis (i.e.
            :math:`ndim < 2`), the shape of output is ``(2, N)`` (``N`` is the
            size of the input variable).

    .. admonition:: Example

        >>> x1 = np.array((1, 2, 3))
        >>> x1.shape
        (3,)
        >>> x2 = np.array((2, 3, 4))
        >>> x2.shape
        (3,)
        >>> y = F.vstack((x1, x2))
        >>> y.shape
        (2, 3)
        >>> y.array
        array([[1, 2, 3],
               [2, 3, 4]])
        >>> x1 = np.arange(0, 12).reshape(3, 4)
        >>> x1.shape
        (3, 4)
        >>> x1
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> x2 = np.arange(12, 20).reshape(2, 4)
        >>> x2.shape
        (2, 4)
        >>> x2
        array([[12, 13, 14, 15],
               [16, 17, 18, 19]])
        >>> y = F.vstack([x1, x2])
        >>> y.shape
        (5, 4)
        >>> y.array
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11],
               [12, 13, 14, 15],
               [16, 17, 18, 19]])

    """

    return Vstack().apply((xs))[0]
