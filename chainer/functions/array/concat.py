import numpy
import six

import chainer
from chainer import backend
from chainer.backends import intel64
from chainer import function_node
from chainer.utils import type_check
import chainerx


class Concat(function_node.FunctionNode):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be int')

        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.make_variable(self.axis, 'axis'))

        type_check.expect(
            -in_types[0].ndim <= self.axis,
            self.axis < in_types[0].ndim
        )
        ndim = type_check.eval(in_types[0].ndim)
        axis = self.axis % ndim
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in six.moves.range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(xs, (4,))):
            # iDeep implementation
            return self._forward_ideep(xs)

        # Generic implementation
        xp = backend.get_array_module(*xs)
        return xp.concatenate(xs, self.axis),

    def forward_chainerx(self, xs):
        return chainerx.concatenate(xs, self.axis),

    def _forward_ideep(self, xs):
        xs_mdarray = intel64.ideep.mdarrayVector()
        for x in xs:
            xs_mdarray.push_back(intel64.ideep.array(x))
        ndim = xs[0].ndim
        axis = self.axis % ndim
        return intel64.ideep.concat.Forward(xs_mdarray, axis),

    def backward(self, indexes, grad_outputs):
        if len(self.inputs) == 1:
            return grad_outputs

        sizes = numpy.array(
            [v.shape[self.axis] for v in self.inputs[:-1]]
        ).cumsum()
        gx, = grad_outputs
        return chainer.functions.split_axis(gx, sizes, self.axis)


def concat(xs, axis=1):
    """Concatenates given variables along an axis.

    Args:
        xs (tuple of :class:`~chainer.Variable` or :ref:`ndarray`):
            Input variables to be concatenated. The variables must have the \
            same shape, except in the dimension corresponding to axis.
        axis (int): The axis along which the arrays will be joined. Default \
            is 1.

    Returns:
        ~chainer.Variable: The concatenated variable.

    .. admonition:: Example

        >>> x = np.arange(0, 12).reshape(3, 4)
        >>> x
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> y = np.arange(0, 3).reshape(3, 1)
        >>> y
        array([[0],
               [1],
               [2]])
        >>> z = F.concat((x, y), axis=1)
        >>> z.array
        array([[ 0,  1,  2,  3,  0],
               [ 4,  5,  6,  7,  1],
               [ 8,  9, 10, 11,  2]])

    """
    y, = Concat(axis).apply(xs)
    return y
