import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Vstack(function.Function):

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
        self.retain_inputs(())
        self._in_shapes = [x.shape for x in xs]
        xp = cuda.get_array_module(*xs)
        return xp.vstack(xs),

    def backward(self, xs, gy):
        if len(self._in_shapes) == 1:
            if len(self._in_shapes[0]) <= 1:
                return gy[0].reshape(self._in_shapes[0]),
            return gy

        xp = cuda.get_array_module(*gy)

        if len(self._in_shapes[0]) <= 1:
            ys = xp.vsplit(gy[0], len(xs))
            return [y.reshape(self._in_shapes[0]) for y in ys]
        else:
            sizes = numpy.array(
                [shape[0] for shape in self._in_shapes[:-1]]).cumsum()
            return xp.vsplit(gy[0], sizes)


def vstack(xs):
    """Concatenate variables vertically (row wise).

    Args:
        xs (list of chainer.Variable): Variables to be concatenated.

    Returns:
        ~chainer.Variable: Output variable.

    """

    return Vstack()(*xs)
