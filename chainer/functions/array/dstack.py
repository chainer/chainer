import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class Dstack(function_node.FunctionNode):

    """Concatenate multiple tensors along third axis (depth wise)."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = type_check.eval(in_types[0].ndim)
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 2:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in six.moves.range(0, ndim):
                if d == 2:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)
        return xp.dstack(xs),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ndim = len(self.inputs[0].shape)
        if len(self.inputs) == 1:
            if ndim <= 2:
                return gy.reshape(self.inputs[0].shape),
            return gy,

        if ndim <= 2:
            gxs = chainer.functions.split_axis(gy, len(self.inputs), axis=2)
            return [gx.reshape(self.inputs[0].shape) for gx in gxs]

        sizes = numpy.array([x.shape[2] for x in self.inputs[:-1]]).cumsum()
        return chainer.functions.split_axis(gy, sizes, axis=2)


def dstack(xs):
    """Concatenate variables along third axis (depth wise).

    Args:
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variables to be concatenated. The variables must have the
            same ``ndim``. When the variables have the third axis (i.e.
            :math:`ndim \\geq 3`), the variables must have the same shape
            along all but the third axis. When the variables do not have the
            third axis(i.e. :math:`ndim < 3`), the variables must have the
            same shape.

    Returns:
        ~chainer.Variable:
            Output variable. When the input variables have the third axis
            (i.e. :math:`ndim \\geq 3`), the shapes of inputs and output are
            the same along all but the third axis. The length of third axis
            is the sum of the lengths of inputs' third axis.
            When the shape of variables are ``(N1, N2)`` (i.e.
            :math:`ndim = 2`), the shape of output is ``(N1, N2, 2)``. When
            the shape of variables are ``(N1,)`` (i.e. :math:`ndim = 1`), the
            shape of output is ``(1, N1, 2)``. When the shape of variables are
            ``()`` (i.e. :math:`ndim = 0`), the shape of output is
            ``(1, 1, 2)``.


    .. admonition:: Example

        >>> x1 = np.array((1, 2, 3))
        >>> x1.shape
        (3,)
        >>> x2 = np.array((2, 3, 4))
        >>> x2.shape
        (3,)
        >>> y = F.dstack((x1, x2))
        >>> y.shape
        (1, 3, 2)
        >>> y.data
        array([[[1, 2],
                [2, 3],
                [3, 4]]])

        >>> x1 = np.arange(0, 6).reshape(3, 2)
        >>> x1.shape
        (3, 2)
        >>> x1
        array([[0, 1],
               [2, 3],
               [4, 5]])
        >>> x2 = np.arange(6, 12).reshape(3, 2)
        >>> x2.shape
        (3, 2)
        >>> x2
        array([[ 6,  7],
               [ 8,  9],
               [10, 11]])
        >>> y = F.dstack([x1, x2])
        >>> y.shape
        (3, 2, 2)
        >>> y.data
        array([[[ 0,  6],
                [ 1,  7]],
        <BLANKLINE>
               [[ 2,  8],
                [ 3,  9]],
        <BLANKLINE>
               [[ 4, 10],
                [ 5, 11]]])

        >>> x1 = np.arange(0, 12).reshape(3, 2, 2)
        >>> x2 = np.arange(12, 18).reshape(3, 2, 1)
        >>> y = F.dstack([x1, x2])
        >>> y.shape
        (3, 2, 3)
        >>> y.data
        array([[[ 0,  1, 12],
                [ 2,  3, 13]],
        <BLANKLINE>
               [[ 4,  5, 14],
                [ 6,  7, 15]],
        <BLANKLINE>
               [[ 8,  9, 16],
                [10, 11, 17]]])

    """

    return Dstack().apply((xs))[0]
